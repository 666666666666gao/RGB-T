"""Collect mixed causal search/action supervision on LasHeR TRAIN partitions.

Prefixes use frozen full GOLA/C1 or a frozen learned ABC policy. Query inputs
are copied before future images. GT supplies supervision, never online actions.
"""
import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .bounded_recovery import BoundedRecoveryTracker
from .candidate_learning import FrozenCandidateExtractor, CandidateQualityHead, selection_scores
from .collect_rollouts import observe_actions, iou
from .collect_temporal import paths, advance_prefix
from .evaluate_online import read_pair
from .temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, reverse_siamfc_cropping_params
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider


REGIONS = ('original', 'anchor_local', 'anchor_widened', 'anchor_fixed_offset',
           'anchor_motion_0', 'anchor_motion_1', 'anchor_motion_2')


def instance_pool(tokens, boxes):
    """Area-weighted patch pooling over predicted boxes in normalized crop space."""
    grid = torch.arange(16, device=tokens.device, dtype=torch.float32) / 16
    y, x = torch.meshgrid(grid, grid, indexing='ij')
    left = torch.stack((x.flatten(), y.flatten()), -1)
    extent = (torch.minimum(boxes[..., None, 2:], left + 1 / 16)
              - torch.maximum(boxes[..., None, :2], left)).clamp(min=0)
    weights = extent.prod(-1)
    assert (weights.sum(-1) > 0).all()
    return torch.einsum('bkn,bnd->bkd', weights, tokens.float()) / weights.sum(-1, keepdim=True)


class InstanceExtractor(FrozenCandidateExtractor):
    @torch.no_grad()
    def extract_with_output(self, batch):
        candidates, output = super().extract_with_output(batch)
        candidates['instance_features'] = torch.stack([
            instance_pool(self.base.patch_embed(batch['x'][:, channel]), candidates['boxes'])
            for channel in (slice(0, 3), slice(3, 6))], dim=2)
        return candidates, output


def history_arrays(context, anchor, capacity):
    row = {'anchor_features': anchor.copy(),
           'history_descriptors': np.broadcast_to(anchor, (capacity, 2, 768)).astype(np.float16).copy(),
           'history_instance_descriptors': np.broadcast_to(anchor, (capacity, 2, 768)).astype(np.float16).copy(),
           'history_evidence': np.zeros((capacity, 10), dtype=np.float32),
           'history_quality': np.zeros(capacity, dtype=np.float32),
           'history_boxes': np.broadcast_to(context['initial'], (capacity, 4)).astype(np.float32).copy(),
           'history_frames': np.zeros(capacity, dtype=np.int64),
           'history_write': np.zeros(capacity, dtype=bool),
           'history_valid': np.zeros(capacity, dtype=bool)}
    for slot, entry in enumerate(context['history'], capacity - len(context['history'])):
        row['history_descriptors'][slot] = anchor if entry['descriptor'] is None else entry['descriptor']
        row['history_instance_descriptors'][slot] = anchor if entry['instance'] is None else entry['instance']
        for field in ('evidence', 'quality', 'box', 'frame', 'write'):
            key = {'box': 'boxes', 'frame': 'frames'}.get(field, field)
            row['history_' + key][slot] = entry[field]
        row['history_valid'][slot] = True
    assert row['history_frames'][-1] < context['query'] and row['history_valid'][-1]
    return row


def decode(candidates, transform, image):
    boxes = (candidates['boxes'][0] * 224.).double().cpu().numpy()
    boxes = apply_siamfc_cropping_to_boxes(boxes, reverse_siamfc_cropping_params(transform))
    for box in boxes:
        bbox_clip_to_image_boundary_(box, np.array((image.shape[-1], image.shape[-2])))
    assert np.isfinite(boxes).all()
    return boxes


@torch.inference_mode()
def collect_batch(jobs, dataset, extractor, head, motion, device, args, prefix_model=None,
                  future_policy='c1', prefix_write_verification='identity'):
    assert future_policy in ('c1', 'own')
    assert future_policy != 'own' or prefix_model is not None
    contexts = []
    for index, query in jobs:
        sequence = dataset[index]
        initial = sequence[0].get_bounding_box().copy()
        image = read_pair(*paths(sequence, 0), device)
        if prefix_model is None:
            tracker = BoundedRecoveryTracker(extractor, head, image, initial, torch.float16)
        else:
            from .recoverability_tracker import RecoverabilityTracker
            tracker = RecoverabilityTracker(extractor, prefix_model, motion, image, initial, args.prefix_threshold,
                                             write_verification=prefix_write_verification,
                                             search_value=args.prefix_search_value)
        evidence = np.zeros(10, dtype=np.float32)
        evidence[:6] = 1
        history = deque([{'descriptor': None, 'instance': None, 'evidence': evidence,
                          'quality': 1., 'box': initial, 'frame': 0, 'write': False}], maxlen=args.max_prefix)
        contexts.append({'sequence': sequence, 'query': query, 'initial': initial,
                         'tracker': tracker, 'branch': tracker.branches[0], 'history': history})
    for frame in range(1, max(q for _, q in jobs)):
        active = [c for c in contexts if frame < c['query']]
        images = [read_pair(*paths(c['sequence'], frame), device) for c in active]
        if prefix_model is None:
            items = [(c['tracker'], c['branch'], im) for c, im in zip(active, images)]
            observations = ([observe_actions([item], extractor, head)[0] for item in items]
                            if args.serial_c1_prefix else observe_actions(items, extractor, head))
            for context, image, observation in zip(active, images, observations):
                advance_prefix(context, frame, observation, image)
                candidates, _, _, choice = observation
                context['history'][-1]['instance'] = candidates['instance_features'][0, choice].cpu().numpy().copy()
        else:
            for context, image in zip(active, images):
                tracker = context['tracker']
                box = tracker.step(image)
                candidates, quality, _, choice = tracker.last_observation
                assert tracker.frame == frame
                context['branch'] = tracker.branch
                context['history'].append({
                    'descriptor': candidates['modality_features'][0, choice].cpu().numpy().copy(),
                    'instance': candidates['instance_features'][0, choice].cpu().numpy().copy(),
                    'evidence': candidates['evidence'][0, choice].cpu().numpy().copy(),
                    'quality': float(quality[choice]), 'box': box.copy(), 'frame': frame,
                    'write': bool(tracker.last_decision['template_updated'])})
    return collect_contexts(contexts, extractor, head, motion, device, args, prefix_model,
                            future_policy, prefix_write_verification)


@torch.inference_mode()
def collect_contexts(contexts, extractor, head, motion, device, args, prefix_model=None,
                     future_policy='c1', prefix_write_verification='identity'):
    """Label queries from existing causal contexts, preserving the old rollout."""
    assert future_policy in ('c1', 'own') and (future_policy != 'own' or prefix_model is not None)
    images = [read_pair(*paths(c['sequence'], c['query']), device) for c in contexts]
    original = observe_actions([(c['tracker'], c['branch'], im) for c, im in zip(contexts, images)], extractor, head)
    rows = [history_arrays(c, obs[0]['anchor_features'][0].cpu().numpy(), args.max_prefix)
            for c, obs in zip(contexts, original)]
    if prefix_model is not None:
        for row, context in zip(rows, contexts):
            row['prefix_counts'] = np.asarray([context['tracker'].stats[key] for key in
                ('extra_visual_forwards', 'changed_candidate_indices', 'paused_query_writes', 'template_updates')], dtype=np.int64)
    history = {key: torch.from_numpy(np.stack([row[key] for row in rows])).to(device)
               for key in ('anchor_features', 'history_descriptors', 'history_evidence', 'history_quality',
                           'history_boxes', 'history_frames', 'history_write', 'history_valid')}
    anchor, memory, _ = motion.history_memory(history)
    distribution = motion.motion(history['history_boxes'].float(), history['history_frames'], history['history_valid'],
                                  history['history_quality'].float(), anchor, memory)
    return label_contexts(contexts, images, original, rows, distribution, extractor, head, device, args, future_policy)


@torch.inference_mode()
def label_contexts(contexts, images, original, rows, distribution, extractor, head, device, args, future_policy):
    """Shared visual-region and private-future action labeling at a fixed state."""
    reference = distribution['reference'].cpu().numpy().astype(np.float64)
    size = np.maximum(reference[:, 2:] - reference[:, :2], 10.)
    center = reference.reshape(-1, 2, 2).mean(1)
    mean = distribution['means'][:, :, 0, :2].cpu().numpy()
    centers = center[:, None] + mean * size[:, None]
    proposals = np.concatenate((centers - size[:, None] / 2, centers + size[:, None] / 2), -1)
    assert np.isfinite(proposals).all()
    directions = np.asarray(((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)))
    crops, transforms, extra_items, requested_areas = [], [], [], []
    region_observations = [[obs] + [None] * 6 for obs in original]
    for clip, (context, image) in enumerate(zip(contexts, images)):
        box = context['branch'].search_box.copy()
        provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
        provider.initialize(box)
        shift = directions[context['query'] % 8] * (224. / provider.get(context['tracker'].search_size)[0]) * .5
        requests = [(box, 4.), (box, 8.), (box + np.tile(shift, 2), 4.)] + [(b, 4.) for b in proposals[clip]]
        areas = np.zeros(6, dtype=np.float64)
        for region, (search_box, factor) in enumerate(requests, 1):
            provider = SiamFCCroppingParameterSimpleProvider(factor, 10.)
            provider.initialize(search_box)
            requested = provider.get(context['tracker'].search_size)
            crop, _, transform = apply_siamfc_cropping(image, context['tracker'].search_size, requested,
                                                      'bilinear', False, context['tracker'].image_mean)
            areas[region - 1] = float(np.prod(224. / requested[0]))
            # Actual TRAIN witness: padding-only extra crop has no inverse.
            if not np.isfinite(transform).all():
                continue
            crops.append(context['tracker'].normalization(crop / 255.))
            transforms.append(transform)
            extra_items.append((clip, region))
        requested_areas.append(areas)
    extractor.proposal_policy = 'dense_raw5'
    for start in range(0, len(extra_items), args.forward_batch):
        items = extra_items[start:start + args.forward_batch]
        batch = {'z': torch.stack([contexts[i]['tracker'].anchor for i, _ in items]),
                 'd': torch.stack([contexts[i]['tracker'].anchor for i, _ in items]),
                 'x': torch.stack(crops[start:start + len(items)]),
                 'z_feat_mask': torch.cat([contexts[i]['tracker'].anchor_mask for i, _ in items]),
                 'd_feat_mask': torch.cat([contexts[i]['tracker'].anchor_mask for i, _ in items])}
        with torch.autocast('cuda', dtype=torch.float16):
            candidates = extractor(batch)
            logits = head(candidates).float()
            scores = selection_scores(logits, candidates, extractor.window_penalty)
        choices = scores.masked_fill(~candidates['valid'], -torch.inf).argmax(1).tolist()
        for offset, (clip, region) in enumerate(items):
            candidate = {key: value[offset:offset + 1] for key, value in candidates.items()}
            boxes = decode(candidate, transforms[start + offset], images[clip])
            region_observations[clip][region] = (candidate, logits.sigmoid()[offset], boxes, choices[offset])
    extractor.proposal_policy = 'peaks'
    # Copy every decision array before any future image or later GT is read.
    fields = ('features', 'evidence', 'raw_score', 'boxes', 'modality_features', 'instance_features', 'valid')
    for clip, row in enumerate(rows):
        for key in fields:
            shape = original[clip][0][key][0].shape
            row[key] = np.zeros((7, *shape), dtype=original[clip][0][key].cpu().numpy().dtype)
        row['image_boxes'] = np.zeros((7, 5, 4), dtype=np.float32)
        row['c1_quality'] = np.zeros((7, 5), dtype=np.float32)
        row['original_choice'] = np.int64(original[clip][3])
        row['region_executed'] = np.zeros(7, dtype=bool)
        row['requested_extra_crop_area'] = requested_areas[clip]
        row['motion_means'] = distribution['means'][clip].cpu().numpy().copy()
        row['motion_log_weights'] = distribution['log_weights'][clip].cpu().numpy().copy()
        for region, observation in enumerate(region_observations[clip]):
            if observation is None:
                continue
            candidate, quality, boxes, _ = observation
            for key in fields:
                row[key][region] = candidate[key][0].cpu().numpy().copy()
            row['image_boxes'][region] = boxes
            row['c1_quality'][region] = quality.cpu().numpy().copy()
            row['region_executed'][region] = True
    actions = []
    for clip, (context, row) in enumerate(zip(contexts, rows)):
        sequence, query, parent = context['sequence'], context['query'], context['branch']
        row['history_iou'] = np.full(args.max_prefix, -1., dtype=np.float32)
        for slot in np.flatnonzero(row['history_valid']):
            gt = sequence[int(row['history_frames'][slot])].get_bounding_box()
            if np.isfinite(gt).all() and (gt[2:] > gt[:2]).all():
                row['history_iou'][slot] = iou(row['history_boxes'][slot], gt)
        row['motion_targets'] = np.stack([sequence[query + h].get_bounding_box() for h in range(3)]).astype(np.float32)
        row['current_iou'] = np.zeros((7, 5), dtype=np.float32)
        row['action_valid'] = np.zeros((7, 5, 2), dtype=bool)
        row['future_iou'] = np.zeros((7, 5, 2, 3), dtype=np.float32)
        row['wrong_update_fraction'] = np.zeros((7, 5, 2), dtype=np.float32)
        for region, observation in enumerate(region_observations[clip]):
            if observation is None:
                continue
            candidate, quality, boxes, _ = observation
            for choice in np.flatnonzero(row['valid'][region]):
                overlap = iou(boxes[choice], sequence[query].get_bounding_box())
                row['current_iou'][region, choice] = overlap
                writes = float(candidate['raw_score'][0, choice]) > .84
                for pause in range(2 if writes else 1):
                    shadow = copy.copy(context['tracker'])
                    shadow.stats = context['tracker'].stats.copy()
                    shadow.frame = query
                    # Future-own steps append history: each action needs a private deque.
                    if future_policy == 'own':
                        shadow.history = deque(context['tracker'].history, maxlen=8)
                        child, _, _, _ = shadow.accept_observation(
                            images[clip], (candidate, quality, boxes, int(choice)), bool(pause))
                    else:
                        # Unchanged C1 teacher; only choice/query write changes.
                        child = shadow._advance(parent, images[clip], boxes, candidate, quality, int(choice))
                        if pause:
                            child.template, child.mask = parent.template, parent.mask
                            child.pending = deque((e for e in parent.pending if query - e[0] < shadow.window), maxlen=shadow.window)
                    row['action_valid'][region, choice, pause] = True
                    actions.append({'clip': clip, 'region': region, 'choice': int(choice), 'pause': pause,
                                    'tracker': shadow, 'branch': child,
                                    'wrong_updates': int(writes and not pause and overlap < .2)})
    for offset in range(1, 4):
        future_images = [read_pair(*paths(c['sequence'], c['query'] + offset), device) for c in contexts]
        for start in range(0, len(actions), args.forward_batch):
            group = actions[start:start + args.forward_batch]
            if future_policy == 'c1':
                observations = observe_actions([(a['tracker'], a['branch'], future_images[a['clip']]) for a in group], extractor, head)
                for action, observation in zip(group, observations):
                    shadow = action['tracker']
                    shadow.frame += 1
                    candidate, quality, boxes, choice = observation
                    action['branch'] = shadow._advance(action['branch'], future_images[action['clip']], boxes, candidate, quality, choice)
                    clip, region, original_choice, pause = (action[k] for k in ('clip', 'region', 'choice', 'pause'))
                    overlap = iou(action['branch'].box, contexts[clip]['sequence'][contexts[clip]['query'] + offset].get_bounding_box())
                    rows[clip]['future_iou'][region, original_choice, pause, offset - 1] = overlap
                    action['wrong_updates'] += int(float(candidate['raw_score'][0, choice]) > .84 and overlap < .2)
            else:
                for action in group:
                    shadow = action['tracker']
                    clip, region, original_choice, pause = (action[k] for k in ('clip', 'region', 'choice', 'pause'))
                    box = shadow.step(future_images[clip])
                    action['branch'] = shadow.branch
                    overlap = iou(box, contexts[clip]['sequence'][contexts[clip]['query'] + offset].get_bounding_box())
                    rows[clip]['future_iou'][region, original_choice, pause, offset - 1] = overlap
                    action['wrong_updates'] += int(bool(shadow.last_decision['template_updated']) and overlap < .2)
    for action in actions:
        clip, region, choice, pause = (action[k] for k in ('clip', 'region', 'choice', 'pause'))
        rows[clip]['wrong_update_fraction'][region, choice, pause] = action['wrong_updates'] / 4
    for row in rows:
        assert all(np.isfinite(value).all() for value in row.values())
        assert row['action_valid'][0, row['original_choice'], 0]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--partition', choices=['train', 'validation'], required=True)
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    p.add_argument('--head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--motion-run', default='/data/gb/outputs/abc_joint_v1_seed42')
    p.add_argument('--prefix-model', help='Frozen ABC checkpoint for actual learned-policy prefixes; default is unchanged C1.')
    p.add_argument('--future-policy', choices=('c1', 'own'), default='c1',
                   help='Use the frozen C1 teacher or the supplied ABC policy for three future steps.')
    p.add_argument('--prefix-write-verification', choices=('identity', 'action'), default='identity')
    p.add_argument('--prefix-search-value', choices=('weighted', 'gross'), default='weighted',
                   help='Execute the supplied prefix/future policy with the deployed search trigger.')
    p.add_argument('--serial-c1-prefix', action='store_true', help='Run only C1 prefix frames individually to match learned-policy prefix execution.')
    p.add_argument('--jobs-file', help='Explicit sequence/query JSON jobs from the chosen TRAIN partition.')
    p.add_argument('--output', required=True)
    p.add_argument('--clips', type=int, default=256)
    p.add_argument('--batch-clips', type=int, default=16)
    p.add_argument('--forward-batch', type=int, default=64)
    p.add_argument('--max-prefix', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    assert args.clips > 0 and args.batch_clips > 0 and args.forward_batch > 0 and args.max_prefix >= 8
    assert args.future_policy != 'own' or args.prefix_model
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    eligible = []
    for index in range(len(dataset)):
        sequence = dataset[index]
        if sequence.get_name() not in split[args.partition]:
            continue
        boxes = sequence.get_all_bounding_boxes()
        valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
        queries = [t for t in range(1, min(args.max_prefix + 1, len(boxes) - 3)) if valid[t:t + 4].all()]
        if valid[0] and queries:
            eligible.append((index, queries))
    rng = np.random.default_rng(args.seed)
    jobs = []
    for _ in range(args.clips):
        index, queries = eligible[int(rng.integers(len(eligible)))]
        jobs.append((index, int(rng.choice(queries))))
    if args.jobs_file:
        requested_jobs = json.loads(Path(args.jobs_file).read_text())['jobs']
        eligible_names = {dataset[index].get_name(): (index, set(queries)) for index, queries in eligible}
        jobs = []
        for job in requested_jobs:
            index, queries = eligible_names[job['sequence']]
            assert job['query_frame'] in queries
            jobs.append((index, job['query_frame']))
        assert jobs and len(set(jobs)) == len(jobs)
        args.clips = len(jobs)
    c1 = torch.load(args.head, map_location='cpu', weights_only=False)
    assert c1['module'] == 'C1_candidate_quality' and c1['args']['candidates'] == 5
    extractor = InstanceExtractor(args.pretrained, c1['args']['candidates'], .45, c1['args']['nms_iou']).to(device)
    head = CandidateQualityHead(c1['args']['hidden']).to(device)
    head.load_state_dict(c1['head'], strict=True)
    head.eval().requires_grad_(False)
    motion_config = json.loads((Path(args.motion_run) / 'config.json').read_text())
    checkpoint = torch.load(Path(args.motion_run) / 'last.pth', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 30 and motion_config['c1_head'] == args.head
    motion = TemporalModules(c1, motion_config['slots'], motion_config['motion_history'], motion_config['modes'], checkpoint['horizon']).to(device)
    motion.load_state_dict(checkpoint['model'], strict=True)
    motion.eval().requires_grad_(False)
    prefix_model = None
    if args.prefix_model:
        from .recoverability_modules import RecoverabilityModules
        prefix_checkpoint = torch.load(args.prefix_model, map_location='cpu', weights_only=False)
        assert prefix_checkpoint['module'] in ('ABC_recoverability', 'ABC_candidate_relations', 'ABC_post_search_relations')
        prefix_model = RecoverabilityModules(c1, candidate_relations=prefix_checkpoint['module'] in ('ABC_candidate_relations', 'ABC_post_search_relations'),
                                            post_search_bidirectional=prefix_checkpoint['module'] == 'ABC_post_search_relations').to(device)
        prefix_model.load_state_dict(prefix_checkpoint['model'], strict=True)
        prefix_model.eval().requires_grad_(False)
        args.prefix_threshold = prefix_checkpoint['args']['threshold']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'jobs': [{'sequence': dataset[i].get_name(), 'query_frame': q} for i, q in jobs],
                           'regions': REGIONS, 'source': 'LasHeR training split only; existing881/98 disjoint partition',
                           'prefix_policy': 'frozen full GOLA/C1 peaks, actual predicted history through query-1',
                           'prefix_execution': 'per-sequence' if prefix_model is not None or args.serial_c1_prefix else 'batch-clips',
                           'original_proposal_policy': 'unchanged frozen C1 five peaks',
                           'extra_proposal_policy': 'protected first anchor, dense_raw5 preserving region Hann winner; up to5 per region',
                           'instance_features': 'area-overlap weighted pre-attention RGB/TIR patches; anchor uses foreground region',
                           'future_policy': 'same frozen C1 peaks; all actions start same parent; pause changes only query template write; subsequent raw>.84 updates retained',
                           'future_policy_mode': args.future_policy,
                           'future_horizon': 3, 'decision_inputs_copied_before_future_decode': True,
                           'history_capacity_covers_prefix': True, 'history_descriptor_storage': 'float16; motion inputs explicitly cast float32',
                           'sampling': 'uniform sequence then valid query, no GT failure selection; includes correct/incorrect/long-prefix states',
                           'pretrained_load': extractor.load_receipt}
    if prefix_model is not None:
        config['prefix_policy'] = 'frozen learned ABC; actual one-extra-region decisions and verified query writes through query-1'
        config['prefix_checkpoint_epoch'] = prefix_checkpoint['epoch']
        config['prefix_model_family'] = prefix_checkpoint['module']
        config['prefix_counts_columns'] = ['extra_visual_forwards', 'changed_candidate_indices', 'paused_query_writes', 'template_updates']
    if args.future_policy == 'own':
        config['future_policy'] = 'same frozen ABC checkpoint as prefix; query commits deployed template, identity and motion state; three causal step calls with private per-action histories; future GT only labels'
        config['future_checkpoint_epoch'] = prefix_checkpoint['epoch']
        config['future_execution'] = 'per-action serial, identical deployed step and write verification'
    if args.jobs_file:
        config['sampling'] = 'explicit TRAIN partition jobs; GT may choose supervision states, never online decisions'
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    rows, started = [], time.perf_counter()
    for start in range(0, len(jobs), args.batch_clips):
        rows.extend(collect_batch(jobs[start:start + args.batch_clips], dataset, extractor, head, motion, device, args,
                                  prefix_model, args.future_policy, args.prefix_write_verification))
        progress = {'completed_clips': len(rows), 'clips': len(jobs), 'elapsed_seconds': time.perf_counter() - started,
                    'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20}
        (out / 'progress.json').write_text(json.dumps(progress))
        print('PROGRESS', json.dumps(progress), flush=True)
    np.savez_compressed(out / 'samples.npz', **{k: np.stack([row[k] for row in rows]) for k in rows[0]})
    current = np.stack([row['current_iou'] for row in rows])
    valid = np.stack([row['valid'] for row in rows])
    choices = np.asarray([row['original_choice'] for row in rows])
    original_iou = current[np.arange(len(rows)), 0, choices]
    original_recall = np.where(valid[:, 0], current[:, 0], -1).max(-1) >= .5
    receipt = {'completed': True, 'clips': len(rows), 'partition': args.partition,
               'valid_actions': int(sum(row['action_valid'].sum() for row in rows)),
               'requested_extra_crops': 6 * len(rows), 'executed_extra_crops': int(sum(row['region_executed'][1:].sum() for row in rows)),
               'original_correct_queries': int((original_iou >= .5).sum()), 'original_failed_queries': int((original_iou < .2).sum()),
               'original_candidate_present_queries': int(original_recall.sum()),
               'query_frames_after256': sum(q > 256 for _, q in jobs),
               'decision_input_contains_future': False, 'official_tracking_accuracy': False,
               'elapsed_seconds': time.perf_counter() - started, 'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20}
    if prefix_model is not None:
        receipt['actual_prefix_counts_sum'] = np.stack([row['prefix_counts'] for row in rows]).sum(0).tolist()
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
