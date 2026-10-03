"""Replay TRAIN candidate-missing queries; diagnose dense boxes and peak selection."""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bounded_recovery import BoundedRecoveryTracker
from .candidate_learning import FrozenCandidateExtractor, CandidateQualityHead, box_iou, selection_scores
from .collect_rollouts import observe_actions, iou
from .collect_temporal import paths, advance_prefix
from .evaluate_online import read_pair
from .temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, reverse_siamfc_cropping_params
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider


def select_peaks(boxes, peaks, scores, winner, budget, nms_iou):
    selected = [winner]
    order = scores.masked_fill(~peaks, -torch.inf).argsort(descending=True)
    for candidate in order.tolist():
        if not bool(peaks[candidate]):
            break
        if all(max(abs(candidate // 16 - j // 16), abs(candidate % 16 - j % 16)) >= 2 for j in selected) and bool((box_iou(boxes[selected], boxes[candidate]) < nms_iou).all()):
            selected.append(candidate)
        if len(selected) == budget:
            break
    return selected


@torch.inference_mode()
def audit_batch(jobs, dataset, extractor, head, device):
    contexts = []
    for job in jobs:
        sequence = dataset[job['index']]
        initial = sequence[0].get_bounding_box().copy()
        tracker = BoundedRecoveryTracker(extractor, head, read_pair(*paths(sequence, 0), device), initial, torch.float16)
        contexts.append({'sequence': sequence, 'query': job['query_frame'], 'tracker': tracker,
                         'branch': tracker.branches[0], 'history': deque(maxlen=256)})
    for frame in range(1, max(c['query'] for c in contexts)):
        active = [c for c in contexts if frame < c['query']]
        images = [read_pair(*paths(c['sequence'], frame), device) for c in active]
        observations = observe_actions([(c['tracker'], c['branch'], image) for c, image in zip(active, images)], extractor, head)
        for context, image, observation in zip(active, images, observations):
            advance_prefix(context, frame, observation, image)
    images = [read_pair(*paths(c['sequence'], c['query']), device) for c in contexts]
    crops, params = [], []
    for context, image in zip(contexts, images):
        tracker = context['tracker']
        provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
        provider.initialize(context['branch'].search_box)
        crop, _, transform = apply_siamfc_cropping(image, tracker.search_size, provider.get(tracker.search_size), 'bilinear', False, tracker.image_mean)
        crops.append(tracker.normalization(crop / 255.))
        params.append(transform)
    batch = {'z': torch.stack([c['tracker'].anchor for c in contexts]),
             'd': torch.stack([c['branch'].template for c in contexts]), 'x': torch.stack(crops),
             'z_feat_mask': torch.cat([c['tracker'].anchor_mask for c in contexts]),
             'd_feat_mask': torch.cat([c['branch'].mask for c in contexts])}
    with torch.autocast('cuda', dtype=torch.float16):
        candidates, output = extractor.extract_with_output(batch)
        logits = head(candidates).float()
        scores = selection_scores(logits, candidates, extractor.window_penalty)
    choices = scores.masked_fill(~candidates['valid'], -torch.inf).argmax(1).tolist()
    raw = output['score_map'].float().sigmoid().flatten(1)
    ranked = raw * (1 - extractor.window_penalty) + extractor.window * extractor.window_penalty
    peaks = (output['score_map'] == F.max_pool2d(output['score_map'].unsqueeze(1), 3, 1, 1).squeeze(1)).flatten(1)
    dense = output['boxes'].flatten(1, 2).float()
    dense_pixel_boxes = (dense * 224.).double().cpu().numpy()
    search_rows = {}
    if 'motion_proposals' in jobs[0]:
        search_crops, search_params, search_contexts = [], [], []
        empty_searches = []
        directions = np.asarray(((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)))
        for index, (job, context, image) in enumerate(zip(jobs, contexts, images)):
            if not job['diagnose']:
                continue
            box = context['branch'].search_box.copy()
            scale, _ = params[index]
            shift = directions[job['query_frame'] % 8] * (224. / scale) * .5
            policies = [('anchor_local', box, 4.), ('anchor_widened', box, 8.),
                        ('anchor_fixed_offset', box + np.tile(shift, 2), 4.)]
            policies.extend((f'anchor_motion_{mode}', proposal, 4.)
                            for mode, proposal in enumerate(job['motion_proposals']))
            for policy, search_box, factor in policies:
                provider = SiamFCCroppingParameterSimpleProvider(factor, 10.)
                provider.initialize(search_box)
                requested_transform = provider.get(context['tracker'].search_size)
                crop, _, transform = apply_siamfc_cropping(image, context['tracker'].search_size,
                                                          requested_transform,
                                                          'bilinear', False, context['tracker'].image_mean)
                requested_area = float(np.prod(224. / requested_transform[0]))
                if not np.isfinite(transform).all():
                    # Real TRAIN witness: an off-image motion crop is padding only;
                    # no valid inverse mapping exists, so it creates no candidates.
                    empty_searches.append((index, policy, factor, requested_area))
                    continue
                search_crops.append(context['tracker'].normalization(crop / 255.))
                search_params.append(transform)
                search_contexts.append((index, policy, factor))
        search_batch = {'z': torch.stack([contexts[i]['tracker'].anchor for i, _, _ in search_contexts]),
                        'd': torch.stack([contexts[i]['tracker'].anchor for i, _, _ in search_contexts]),
                        'x': torch.stack(search_crops),
                        'z_feat_mask': torch.cat([contexts[i]['tracker'].anchor_mask for i, _, _ in search_contexts]),
                        'd_feat_mask': torch.cat([contexts[i]['tracker'].anchor_mask for i, _, _ in search_contexts])}
        torch.cuda.synchronize(device)
        visual_started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            search_candidates, search_output = extractor.extract_with_output(search_batch)
            search_scores = selection_scores(head(search_candidates).float(), search_candidates, extractor.window_penalty)
        torch.cuda.synchronize(device)
        search_seconds = time.perf_counter() - visual_started
        search_boxes = (search_output['boxes'].flatten(1, 2).float() * 224.).double().cpu().numpy()
        search_raw = search_output['score_map'].float().sigmoid().flatten(1)
        search_ranked = search_raw * (1 - extractor.window_penalty) + extractor.window * extractor.window_penalty
        for search_index, (index, policy, factor) in enumerate(search_contexts):
            transform = search_params[search_index]
            image = images[index]
            boxes = apply_siamfc_cropping_to_boxes(search_boxes[search_index], reverse_siamfc_cropping_params(transform))
            for box in boxes:
                bbox_clip_to_image_boundary_(box, np.array((image.shape[-1], image.shape[-2])))
            # All search images, proposals and forwards are materialized before query GT.
            gt = contexts[index]['sequence'][contexts[index]['query']].get_bounding_box()
            overlap = np.asarray([iou(box, gt) for box in boxes])
            extra_boxes = search_candidates['boxes'][search_index, search_candidates['valid'][search_index]]
            dense_boxes = search_output['boxes'].flatten(1, 2)[search_index].float()
            winner = int(search_ranked[search_index].argmax())
            all_positions = torch.ones_like(search_raw[search_index], dtype=torch.bool)
            dense_hann5 = select_peaks(dense_boxes, all_positions, search_ranked[search_index], winner, 5, extractor.nms_iou)
            dense_raw5 = select_peaks(dense_boxes, all_positions, search_raw[search_index], winner, 5, extractor.nms_iou)
            extra_indices = []
            for candidate in extra_boxes:
                matches = torch.where((dense_boxes == candidate).all(-1))[0]
                assert len(matches)
                extra_indices.append(int(matches[0]))
            extra_choice = int(search_scores[search_index].masked_fill(~search_candidates['valid'][search_index], -torch.inf).argmax())
            center = apply_siamfc_cropping_to_boxes(gt, transform).reshape(2, 2).mean(0)
            extra_best_score = float(search_scores[search_index, extra_choice])
            search_rows.setdefault(index, {})[policy] = {
                'usable_image_crop': True,
                'extra_candidate_count': len(extra_indices), 'extra_top5_oracle_iou': float(overlap[extra_indices].max()),
                'extra_dense256_oracle_iou': float(overlap.max()),
                'extra_dense_hann5_oracle_iou': float(overlap[dense_hann5].max()),
                'extra_dense_raw5_oracle_iou': float(overlap[dense_raw5].max()),
                'extra_dense_hann5_count': len(dense_hann5), 'extra_dense_raw5_count': len(dense_raw5),
                'extra_c1_selected_iou': float(overlap[extra_indices[extra_choice]]),
                'extra_selected_score': extra_best_score,
                'target_center_inside_actual_adjusted_crop': bool(((center >= 0) & (center <= 224)).all()),
                'actual_crop_area_pixels': float(np.prod(224. / transform[0])),
                'visual_input_size': [224, 224], 'extra_visual_forwards_per_query': 1,
                'extra_template_source': 'protected first-frame anchor for both z and d', 'area_factor': factor,
                'all_six_alternatives_batched_visual_seconds': search_seconds,
                'all_six_alternatives_batched_crop_count': len(search_contexts),
                'all_six_alternatives_batched_query_count': len({i for i, _, _ in search_contexts})}
        for index, policy, factor, requested_area in empty_searches:
            search_rows.setdefault(index, {})[policy] = {
                'usable_image_crop': False, 'reason': 'empty image extent; adjusted crop has no finite inverse',
                'extra_candidate_count': 0, 'extra_top5_oracle_iou': 0., 'extra_dense256_oracle_iou': 0.,
                'extra_dense_hann5_oracle_iou': 0., 'extra_dense_raw5_oracle_iou': 0.,
                'extra_dense_hann5_count': 0, 'extra_dense_raw5_count': 0,
                'extra_c1_selected_iou': 0., 'extra_selected_score': None,
                'target_center_inside_actual_adjusted_crop': False,
                'actual_crop_area_pixels': None, 'requested_crop_area_pixels': requested_area,
                'visual_input_size': [224, 224], 'extra_visual_forwards_per_query': 0,
                'extra_template_source': 'protected first-frame anchor for both z and d', 'area_factor': factor,
                'all_six_alternatives_batched_visual_seconds': search_seconds,
                'all_six_alternatives_batched_crop_count': len(search_contexts),
                'all_six_alternatives_batched_query_count': len({i for i, _, _ in search_contexts})}
    rows = []
    for index, (job, context, image, transform) in enumerate(zip(jobs, contexts, images, params)):
        if not job['diagnose']:
            continue
        normalized = dense[index]
        winner = int(ranked[index].argmax())
        policies = {'hann5': select_peaks(normalized, peaks[index], ranked[index], winner, 5, extractor.nms_iou),
                    'raw5': select_peaks(normalized, peaks[index], raw[index], winner, 5, extractor.nms_iou),
                    'hann16': select_peaks(normalized, peaks[index], ranked[index], winner, 16, extractor.nms_iou),
                    'dense_hann5': select_peaks(normalized, torch.ones_like(peaks[index]), ranked[index], winner, 5, extractor.nms_iou),
                    'dense_raw5': select_peaks(normalized, torch.ones_like(peaks[index]), raw[index], winner, 5, extractor.nms_iou)}
        assert len(policies['hann5']) == int(candidates['valid'][index].sum())
        assert torch.equal(normalized[policies['hann5']], candidates['boxes'][index, candidates['valid'][index]])
        slots = policies['hann5'] + [0] * (5 - len(policies['hann5']))
        boxes = apply_siamfc_cropping_to_boxes(dense_pixel_boxes[index], reverse_siamfc_cropping_params(transform))
        for box in boxes:
            bbox_clip_to_image_boundary_(box, np.array((image.shape[-1], image.shape[-2])))
        # GT is used only after the complete prefix and current visual forward.
        gt = context['sequence'][context['query']].get_bounding_box()
        overlaps = np.asarray([iou(box, gt) for box in boxes])
        selected = policies['hann5'][choices[index]]
        original_oracle = float(overlaps[policies['hann5']].max())
        crop_gt = apply_siamfc_cropping_to_boxes(gt, transform)
        center = (crop_gt[:2] + crop_gt[2:]) / 2
        inside = bool(((center >= 0) & (center <= 224)).all())
        row = {key: job[key] for key in ('cache', 'cache_row', 'sequence', 'query_frame')}
        row.update({'replayed_selected_iou': float(overlaps[selected]),
                    'replayed_original5_oracle_iou': float(original_oracle),
                    'replayed_failed': bool(overlaps[selected] < .2),
                    'replayed_missing_candidate': bool(overlaps[selected] < .2 and original_oracle < .5),
                    'target_center_inside_actual_adjusted_crop': inside,
                    'dense256_oracle_iou': float(overlaps.max()),
                    'local_peak_oracle_iou': float(overlaps[peaks[index].cpu().numpy()].max()),
                    'raw5_oracle_iou': float(overlaps[policies['raw5']].max()),
                    'hann16_oracle_iou': float(overlaps[policies['hann16']].max()),
                    'dense_hann5_oracle_iou': float(overlaps[policies['dense_hann5']].max()),
                    'dense_raw5_oracle_iou': float(overlaps[policies['dense_raw5']].max()),
                    'dense_hann5_candidate_count': len(policies['dense_hann5']),
                    'dense_raw5_candidate_count': len(policies['dense_raw5']),
                    'raw5_candidate_count': len(policies['raw5']), 'hann16_candidate_count': len(policies['hann16']),
                    'replayed_valid_candidates': len(policies['hann5']), 'cache_valid_candidates': int(job['valid'].sum()),
                    'cache_selected_iou': job['cached_iou'],
                    'cache_choice_slot_equal': choices[index] == job['original_choice'],
                    'cache_features_max_abs_difference': float(np.abs(candidates['features'][index].cpu().numpy() - job['features']).max()),
                    'cache_original5_slots_max_abs_difference_pixels': float(np.abs(boxes[slots].astype(np.float32) - job['image_boxes']).max())})
        if search_rows:
            row['extra_search'] = search_rows[index]
            row['motion_top_probability_mode'] = job['motion_top_probability_mode']
            row['original_actual_crop_area_pixels'] = float(np.prod(224. / transform[0]))
            for policy in row['extra_search'].values():
                policy['union_original5_plus_extra5_oracle_iou'] = max(original_oracle, policy['extra_top5_oracle_iou'])
                policy['union_candidate_count'] = len(policies['hann5']) + policy['extra_candidate_count']
                for dense_policy in ('dense_hann5', 'dense_raw5'):
                    policy[f'union_original5_plus_{dense_policy}_oracle_iou'] = max(original_oracle, policy[f'extra_{dense_policy}_oracle_iou'])
                policy['union_score_selected_iou'] = (policy['extra_c1_selected_iou']
                                                      if policy['usable_image_crop'] and policy['extra_selected_score'] > float(scores[index, choices[index]])
                                                      else float(overlaps[selected]))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caches', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help='M0 only; 0 replays every sampled candidate-missing query')
    parser.add_argument('--search-training-run', help='Also audit six anchor-based extra-region alternatives using this frozen ABC last checkpoint.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    source = json.loads((Path(args.caches[0]) / 'config.json').read_text())
    split = json.loads(Path(source['split']).read_text())
    assert not set(split['train']) & set(split['validation'])
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(source['root'], source['cache'])
    motion_model = None
    if args.search_training_run:
        training = Path(args.search_training_run)
        assert (training / 'training_and_audit_completed.txt').is_file()
        model_config = json.loads((training / 'config.json').read_text())
        checkpoint = torch.load(training / 'last.pth', map_location='cpu', weights_only=False)
        assert model_config['c1_head'] == source['head'] and checkpoint['epoch'] == 30
        c1 = torch.load(source['head'], map_location='cpu', weights_only=False)
        motion_model = TemporalModules(c1, model_config['slots'], model_config['motion_history'], model_config['modes'], checkpoint['horizon'])
        motion_model.load_state_dict(checkpoint['model'], strict=True)
        motion_model.eval().requires_grad_(False)
    indices = {dataset[i].get_name(): i for i in range(len(dataset))}
    groups, selected_queries = [], 0
    for cache in map(Path, args.caches):
        config = json.loads((cache / 'config.json').read_text())
        assert json.loads((cache / 'completion.json').read_text())['completed']
        assert all(config[key] == source[key] for key in ('root', 'cache', 'split', 'pretrained', 'head', 'partition'))
        assert config['partition'] in ('train', 'validation')
        with np.load(cache / 'samples.npz') as archive:
            overlaps, valid, choices = archive['current_iou'], archive['valid'], archive['original_choice']
            missing = (overlaps[np.arange(len(choices)), choices] < .2) & (np.where(valid, overlaps, -1).max(1) < .5)
            features, image_boxes = archive['features'], archive['image_boxes']
        selected = np.flatnonzero(missing)
        if args.limit:
            selected = selected[:max(0, args.limit - selected_queries)]
        selected_queries += len(selected)
        motion_proposals, mode_choices = {}, {}
        if motion_model is not None and len(selected):
            fields = ('anchor_features', 'history_descriptors', 'history_evidence', 'history_quality',
                      'history_boxes', 'history_frames', 'history_write', 'history_valid')
            with np.load(cache / 'samples.npz') as archive:
                data = {key: torch.from_numpy(archive[key][selected].copy()) for key in fields}
            with torch.inference_mode():
                anchor, memory, _ = motion_model.history_memory(data)
                distribution = motion_model.motion(data['history_boxes'].float(), data['history_frames'], data['history_valid'],
                                                   data['history_quality'].float(), anchor, memory)
            reference = distribution['reference'].numpy().astype(np.float64)
            assert np.array_equal(reference, data['history_boxes'][:, -1].numpy())
            size = np.maximum(reference[:, 2:] - reference[:, :2], 10.)
            center = (reference[:, :2] + reference[:, 2:]) / 2
            proposal_center = center[:, None] + distribution['means'][:, :, 0, :2].numpy() * size[:, None]
            proposals = np.concatenate((proposal_center - size[:, None] / 2, proposal_center + size[:, None] / 2), -1)
            assert proposals.shape[1:] == (3, 4) and np.isfinite(proposals).all()
            motion_proposals = dict(zip(map(int, selected), proposals))
            mode_choices = dict(zip(map(int, selected), distribution['log_weights'].argmax(-1).tolist()))
        selected = set(map(int, selected))
        batch_size = config['batch_clips']
        for start in range(0, len(config['jobs']), batch_size):
            stop = min(start + batch_size, len(config['jobs']))
            if not selected.intersection(range(start, stop)):
                continue
            jobs = []
            for row in range(start, stop):
                job = config['jobs'][row]
                assert job['sequence'] in split[source['partition']]
                jobs.append({**job, 'cache': str(cache), 'cache_row': row, 'index': indices[job['sequence']],
                             'cached_iou': float(overlaps[row, choices[row]]), 'original_choice': int(choices[row]),
                             'features': features[row].copy(), 'image_boxes': image_boxes[row].copy(), 'valid': valid[row].copy(),
                             'diagnose': row in selected})
                if motion_model is not None:
                    jobs[-1]['motion_proposals'] = motion_proposals[row] if row in selected else None
                    jobs[-1]['motion_top_probability_mode'] = mode_choices[row] if row in selected else None
            groups.append(jobs)
    assert groups
    c1 = torch.load(source['head'], map_location='cpu', weights_only=False)
    extractor = FrozenCandidateExtractor(source['pretrained'], c1['args']['candidates'], .45, c1['args']['nms_iou']).to(device)
    assert extractor.candidates == 5
    head = CandidateQualityHead(c1['args']['hidden']).to(device)
    head.load_state_dict(c1['head'], strict=True)
    head.eval().requires_grad_(False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'partition': source['partition'], 'sample': 'all sampled C1 failed queries without candidate IoU>=.5; GT-selected offline diagnosis',
                'causal': 'first GT initializes; predicted C1 prefix; no future images; query GT only scores after forward',
                'precision': 'original float16 forward and float32-multiply-before-double decoding; actual adjusted crop',
                'comparison': 'same causal search; hann5/raw5 equal proposal budget; hann16/dense256 are larger-budget diagnostic upper bounds',
                'raw5': 'preserves original Hann winner, ranks remaining spatial peaks by raw score with original NMS',
                'dense_hann5_and_dense_raw5': 'same five-proposal budget and original Hann winner, spacing and NMS; rank all256 positions instead of local peaks; no extra visual forward or GT-conditioned proposal',
                'cache_replay': 'full original batch groups and order preserve causal batch shapes; fixed five slots, valid counts and actual feature/box/choice differences are recorded',
                'scope': 'read-only TRAIN/validation GPU replay, no optimizer, no deployed ABC tracker change'}
    if motion_model is not None:
        protocol['extra_search'] = ('protected-anchor local/factor8 widened/fixed half-crop offset/query%8 and three frozen causal motion means; '
                                    'each alternative adds one 224x224 visual crop and up to5 candidates to unchanged original5; '
                                    'all three motion modes together cost four regions/up to20 proposals; '
                                    'factor8 crop has larger physical area, not equal-area; six alternatives are offline batched, not deployed FPS; '
                                    'padding-only crops with no finite adjusted inverse are explicitly unexecuted, generate zero candidates, '
                                    'and retain requested area while actual area is undefined; executed crop counts reflect this; '
                                    'query GT is only a post-forward recall label; failure-selected sample cannot measure harms on original correct frames')
    (output / 'config.json').write_text(json.dumps(vars(args) | {'protocol': protocol, 'pretrained_load': extractor.load_receipt}, indent=2))
    started, rows = time.perf_counter(), []
    for jobs in groups:
        rows.extend(audit_batch(jobs, dataset, extractor, head, device))
        print(json.dumps({'completed_queries': len(rows), 'queries': selected_queries, 'elapsed_seconds': time.perf_counter() - started}), flush=True)
    report = {'completed': True, 'official_tracking_accuracy': False, 'm0_only': bool(args.limit), 'protocol': protocol,
              'sampled_queries': len(rows), 'unique_sequence_query_pairs': len({(r['sequence'], r['query_frame']) for r in rows}),
              'replayed_original_groups': len(groups), 'replayed_prefix_contexts': sum(map(len, groups)),
              'elapsed_seconds': time.perf_counter() - started, 'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
              'rows': rows}
    report['summary'] = {
        'queries': len(rows),
        'cache_exact_features_boxes_valid_counts_and_choice': sum(
            r['cache_features_max_abs_difference'] == r['cache_original5_slots_max_abs_difference_pixels'] == 0
            and r['cache_valid_candidates'] == r['replayed_valid_candidates'] and r['cache_choice_slot_equal'] for r in rows),
        'replayed_missing_candidate': sum(r['replayed_missing_candidate'] for r in rows),
        'target_center_inside_actual_adjusted_original_crop': sum(r['target_center_inside_actual_adjusted_crop'] for r in rows),
        'local_correct_candidate_query_counts': {policy: sum(r[policy + '_oracle_iou'] >= .5 for r in rows)
                                                for policy in ('replayed_original5', 'dense256', 'local_peak', 'raw5', 'hann16', 'dense_hann5', 'dense_raw5')}}
    if motion_model is not None:
        report['summary']['extra_search'] = {
            policy: {'queries': len(rows), 'requested_extra_visual_crops': len(rows),
                     'executed_extra_visual_crops': sum(r['extra_search'][policy]['extra_visual_forwards_per_query'] for r in rows),
                     'extra_visual_forwards_per_query': float(np.mean([r['extra_search'][policy]['extra_visual_forwards_per_query'] for r in rows])),
                     'maximum_original_plus_extra_candidates': 10,
                     'union_correct_candidate_queries': sum(r['extra_search'][policy]['union_original5_plus_extra5_oracle_iou'] >= .5 for r in rows),
                     'extra_correct_dense_queries': sum(r['extra_search'][policy]['extra_dense256_oracle_iou'] >= .5 for r in rows),
                     'union_correct_dense_hann5_queries': sum(r['extra_search'][policy]['union_original5_plus_dense_hann5_oracle_iou'] >= .5 for r in rows),
                     'union_correct_dense_raw5_queries': sum(r['extra_search'][policy]['union_original5_plus_dense_raw5_oracle_iou'] >= .5 for r in rows),
                     'target_center_inside_extra_crop_queries': sum(r['extra_search'][policy]['target_center_inside_actual_adjusted_crop'] for r in rows),
                     'union_score_selection_correct_queries': sum(r['extra_search'][policy]['union_score_selected_iou'] >= .5 for r in rows),
                     'mean_original_plus_extra_crop_area_pixels': float(np.mean([
                         r['original_actual_crop_area_pixels'] + r['extra_search'][policy]['actual_crop_area_pixels']
                         for r in rows if r['extra_search'][policy]['usable_image_crop']]))
                         if any(r['extra_search'][policy]['usable_image_crop'] for r in rows) else None,
                     'queries_with_defined_actual_crop_area': sum(r['extra_search'][policy]['usable_image_crop'] for r in rows)}
            for policy in rows[0]['extra_search']}
        report['summary']['top_probability_motion_union_correct_queries'] = sum(
            r['extra_search'][f"anchor_motion_{r['motion_top_probability_mode']}"]['union_original5_plus_extra5_oracle_iou'] >= .5 for r in rows)
        report['summary']['any_three_motion_union_correct_queries'] = sum(
            max(r['extra_search'][f'anchor_motion_{mode}']['union_original5_plus_extra5_oracle_iou'] for mode in range(3)) >= .5 for r in rows)
    (output / 'dense_candidate_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
