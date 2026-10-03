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
    rows = []
    for index, (job, context, image, transform) in enumerate(zip(jobs, contexts, images, params)):
        if not job['diagnose']:
            continue
        normalized = dense[index]
        winner = int(ranked[index].argmax())
        policies = {'hann5': select_peaks(normalized, peaks[index], ranked[index], winner, 5, extractor.nms_iou),
                    'raw5': select_peaks(normalized, peaks[index], raw[index], winner, 5, extractor.nms_iou),
                    'hann16': select_peaks(normalized, peaks[index], ranked[index], winner, 16, extractor.nms_iou)}
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
        original_oracle = overlaps[policies['hann5']].max()
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
                    'raw5_candidate_count': len(policies['raw5']), 'hann16_candidate_count': len(policies['hann16']),
                    'replayed_valid_candidates': len(policies['hann5']), 'cache_valid_candidates': int(job['valid'].sum()),
                    'cache_selected_iou': job['cached_iou'],
                    'cache_choice_slot_equal': choices[index] == job['original_choice'],
                    'cache_features_max_abs_difference': float(np.abs(candidates['features'][index].cpu().numpy() - job['features']).max()),
                    'cache_original5_slots_max_abs_difference_pixels': float(np.abs(boxes[slots].astype(np.float32) - job['image_boxes']).max())})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caches', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help='M0 only; 0 replays every sampled candidate-missing query')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    source = json.loads((Path(args.caches[0]) / 'config.json').read_text())
    split = json.loads(Path(source['split']).read_text())
    assert not set(split['train']) & set(split['validation'])
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(source['root'], source['cache'])
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
                'cache_replay': 'full original batch groups and order preserve causal batch shapes; fixed five slots, valid counts and actual feature/box/choice differences are recorded',
                'scope': 'read-only TRAIN/validation GPU replay, no optimizer, no deployed ABC tracker change'}
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
    (output / 'dense_candidate_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
