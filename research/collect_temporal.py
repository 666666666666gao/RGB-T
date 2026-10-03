"""Collect causal A/B/C inputs and separate TRAIN supervision.

Prefixes are single-path frozen C1 predictions batched across clips. Every
current/past input is copied before decoding any counterfactual future frame.
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
from .candidate_learning import FrozenCandidateExtractor, CandidateQualityHead
from .collect_rollouts import observe_actions, iou
from .evaluate_online import read_pair
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--partition', choices=['train', 'validation'], required=True)
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    p.add_argument('--head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--output', required=True)
    p.add_argument('--clips', type=int, default=1024)
    p.add_argument('--batch-clips', type=int, default=32)
    p.add_argument('--history', type=int, default=256)
    p.add_argument('--max-prefix', type=int, default=256)
    p.add_argument('--horizon', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def paths(sequence, frame):
    return tuple(map(Path, sequence[frame].get_image_path()))


def advance_prefix(context, frame, observation, image):
    candidates, quality, boxes, choice = observation
    tracker = context['tracker']
    tracker.frame = frame
    branch = tracker._advance(context['branch'], image, boxes, candidates, quality, choice)
    branch.selected_streak += 1
    if branch.selected_streak >= tracker.window and branch.pending:
        _, branch.committed_template, branch.committed_mask = branch.pending[-1]
        branch.pending.clear()
    context['branch'] = branch
    context['history'].append({'descriptor': candidates['modality_features'][0, choice].cpu().numpy().copy(),
                               'evidence': candidates['evidence'][0, choice].cpu().numpy().copy(),
                               'quality': float(quality[choice]), 'box': boxes[choice].copy(), 'frame': frame,
                               'write': float(candidates['raw_score'][0, choice]) > .84})


@torch.inference_mode()
def collect_batch(jobs, dataset, extractor, head, device, args):
    contexts = []
    for sequence_index, query in jobs:
        sequence = dataset[sequence_index]
        initial = sequence[0].get_bounding_box().copy()
        image = read_pair(*paths(sequence, 0), device)
        tracker = BoundedRecoveryTracker(extractor, head, image, initial, torch.float16)
        initial_evidence = np.zeros(10, dtype=np.float32)
        initial_evidence[:6] = 1.
        history = deque([{'descriptor': None, 'evidence': initial_evidence, 'quality': 1.,
                          'box': initial.copy(), 'frame': 0, 'write': False}], maxlen=args.history)
        contexts.append({'sequence': sequence, 'query': query, 'tracker': tracker,
                         'branch': tracker.branches[0], 'history': history})
    # Time increases monotonically in each clip. No past GT enters a decision.
    for frame in range(1, max(query for _, query in jobs)):
        active = [c for c in contexts if frame < c['query']]
        images = [read_pair(*paths(c['sequence'], frame), device) for c in active]
        observations = observe_actions([(c['tracker'], c['branch'], image)
                                        for c, image in zip(active, images)], extractor, head)
        for context, image, observation in zip(active, images, observations):
            advance_prefix(context, frame, observation, image)
    images = [read_pair(*paths(c['sequence'], c['query']), device) for c in contexts]
    current = observe_actions([(c['tracker'], c['branch'], image)
                               for c, image in zip(contexts, images)], extractor, head)
    rows, actions = [], []
    for clip, (context, image, observation) in enumerate(zip(contexts, images, current)):
        sequence, query, tracker, parent = (context[k] for k in ('sequence', 'query', 'tracker', 'branch'))
        candidates, quality, boxes, choice = observation
        anchor = candidates['anchor_features'][0].cpu().numpy().copy()
        row = {'features': candidates['features'][0].cpu().numpy().copy(),
               'evidence': candidates['evidence'][0].cpu().numpy().copy(),
               'raw_score': candidates['raw_score'][0].cpu().numpy().copy(),
               'c1_quality': quality.cpu().numpy().copy(),
               'boxes': candidates['boxes'][0].cpu().numpy().copy(),
               'modality_features': candidates['modality_features'][0].cpu().numpy().copy(),
               'anchor_features': anchor, 'image_boxes': boxes.astype(np.float32),
               'valid': candidates['valid'][0].cpu().numpy().copy(), 'original_choice': choice,
               'history_descriptors': np.broadcast_to(anchor, (args.history, 2, 768)).copy(),
               'history_evidence': np.zeros((args.history, 10), dtype=np.float32),
               'history_quality': np.zeros(args.history, dtype=np.float32),
               'history_boxes': np.broadcast_to(sequence[0].get_bounding_box(), (args.history, 4)).astype(np.float32).copy(),
               'history_frames': np.zeros(args.history, dtype=np.int64),
               'history_write': np.zeros(args.history, dtype=bool),
               'history_valid': np.zeros(args.history, dtype=bool)}
        for offset, entry in enumerate(context['history'], args.history - len(context['history'])):
            row['history_descriptors'][offset] = anchor if entry['descriptor'] is None else entry['descriptor']
            row['history_evidence'][offset] = entry['evidence']
            row['history_quality'][offset] = entry['quality']
            row['history_boxes'][offset] = entry['box']
            row['history_frames'][offset] = entry['frame']
            row['history_write'][offset] = entry['write']
            row['history_valid'][offset] = True
        assert row['history_frames'][-1] < query and row['history_valid'][-1]
        rows.append(row)  # All decision inputs have now been materialized.

        # Historical/current GT below only produces training labels.
        row['history_iou'] = np.full(args.history, -1., dtype=np.float32)
        for slot in np.flatnonzero(row['history_valid']):
            target = sequence[int(row['history_frames'][slot])].get_bounding_box()
            if np.isfinite(target).all() and (target[2:] > target[:2]).all():
                row['history_iou'][slot] = iou(row['history_boxes'][slot], target)
        row['motion_targets'] = np.stack([sequence[query + h].get_bounding_box() for h in range(args.horizon)]).astype(np.float32)
        row['current_iou'] = np.zeros(len(row['valid']), dtype=np.float32)
        row['future_iou'] = np.zeros((len(row['valid']), args.horizon), dtype=np.float32)
        row['wrong_update_fraction'] = np.zeros(len(row['valid']), dtype=np.float32)
        for alternative in np.flatnonzero(row['valid']):
            shadow = copy.copy(tracker)
            shadow.stats = tracker.stats.copy()
            shadow.frame = query
            child = shadow._advance(parent, image, boxes, candidates, quality, int(alternative), fork=alternative != choice)
            overlap = iou(child.box, sequence[query].get_bounding_box())
            row['current_iou'][alternative] = overlap
            wrong = int(float(candidates['raw_score'][0, alternative]) > .84 and overlap < .2)
            actions.append({'clip': clip, 'choice': int(alternative), 'tracker': shadow,
                            'branch': child, 'wrong_updates': wrong})
    # Future images are decoded only after all current/past inputs are copied.
    for offset in range(1, args.horizon + 1):
        images = [read_pair(*paths(c['sequence'], c['query'] + offset), device) for c in contexts]
        observations = observe_actions([(a['tracker'], a['branch'], images[a['clip']]) for a in actions], extractor, head)
        for action, observation in zip(actions, observations):
            shadow, clip, choice = action['tracker'], action['clip'], action['choice']
            shadow.frame += 1
            candidates, quality, boxes, selected = observation
            action['branch'] = shadow._advance(action['branch'], images[clip], boxes, candidates, quality, selected)
            context = contexts[clip]
            overlap = iou(action['branch'].box, context['sequence'][context['query'] + offset].get_bounding_box())
            rows[clip]['future_iou'][choice, offset - 1] = overlap
            action['wrong_updates'] += int(float(candidates['raw_score'][0, selected]) > .84 and overlap < .2)
    for action in actions:
        rows[action['clip']]['wrong_update_fraction'][action['choice']] = action['wrong_updates'] / (args.horizon + 1)
    for row in rows:
        assert all(np.isfinite(value).all() for value in row.values())
    return rows


def main():
    args = arguments()
    assert args.history >= args.max_prefix and args.max_prefix >= 1
    assert args.history >= 8 and args.horizon == 3 and args.clips > 0 and args.batch_clips > 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    names, eligible = set(split[args.partition]), []
    for index in range(len(dataset)):
        if dataset[index].get_name() not in names:
            continue
        boxes = dataset[index].get_all_bounding_boxes()
        valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
        if not valid[0]:
            continue
        queries = [t for t in range(1, min(args.max_prefix + 1, len(boxes) - args.horizon))
                   if valid[t:t + args.horizon + 1].all()]
        if queries:
            eligible.append((index, queries))
    rng = np.random.default_rng(args.seed)
    jobs = []
    for _ in range(args.clips):
        index, queries = eligible[int(rng.integers(len(eligible)))]
        jobs.append((index, int(rng.choice(queries))))
    c1 = torch.load(args.head, map_location='cpu', weights_only=False)
    assert c1['module'] == 'C1_candidate_quality'
    extractor = FrozenCandidateExtractor(args.pretrained, c1['args']['candidates'], .45, c1['args']['nms_iou']).to(device)
    head = CandidateQualityHead(c1['args']['hidden']).to(device)
    head.load_state_dict(c1['head'], strict=True)
    head.eval().requires_grad_(False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'jobs': [{'sequence': dataset[i].get_name(), 'query_frame': t} for i, t in jobs],
                           'source': 'LasHeR TRAIN only; original881/98 disjoint C1 split',
                           'prefix_policy': 'single path frozen C1; predictions only, no later GT',
                           'future_policy': 'identical frozen C1 continuation, branch-private templates',
                           'decision_inputs_copied_before_future_decode': True,
                           'modality_descriptors': 'separate pre-attention RGB/TIR patch tokens',
                           'temporal_labels': 'past/current GT IoU; future GT geometry; future rollout IoU/wrong writes',
                           'pretrained_load': extractor.load_receipt, 'amp_dtype': 'float16'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    started, rows = time.perf_counter(), []
    for start in range(0, len(jobs), args.batch_clips):
        rows.extend(collect_batch(jobs[start:start + args.batch_clips], dataset, extractor, head, device, args))
        progress = {'completed_clips': len(rows), 'clips': len(jobs),
                    'elapsed_seconds': time.perf_counter() - started,
                    'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20}
        (out / 'progress.json').write_text(json.dumps(progress, indent=2))
        print('PROGRESS', json.dumps(progress), flush=True)
    np.savez_compressed(out / 'samples.npz', **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
    receipt = {'completed': True, 'clips': len(rows), 'partition': args.partition,
               'valid_actions': int(sum(r['valid'].sum() for r in rows)),
               'elapsed_seconds': time.perf_counter() - started,
               'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'decision_input_contains_future': False, 'official_tracking_accuracy': False}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
