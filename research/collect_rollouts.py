"""Generate C3 counterfactual labels from predicted histories on TRAIN split.

Decision features are saved before future frames are decoded. Future frames and
GT supervise fixed-C1 continuation only; they never become decision inputs.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .bounded_recovery import BoundedRecoveryTracker
from .branch_utility import motion_evidence
from .candidate_learning import FrozenCandidateExtractor, CandidateQualityHead, box_iou, selection_scores
from .evaluate_online import read_pair
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping, apply_siamfc_cropping_to_boxes, reverse_siamfc_cropping_params
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--partition', choices=['train', 'validation'], required=True)
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    p.add_argument('--head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--output', required=True)
    p.add_argument('--clips', type=int, default=128)
    p.add_argument('--batch-clips', type=int, default=8)
    p.add_argument('--min-history', type=int, default=1)
    p.add_argument('--max-history', type=int, default=256)
    p.add_argument('--horizon', type=int, default=3)
    p.add_argument('--pollution-weight', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def observe_actions(items, extractor, head):
    crops, params = [], []
    for tracker, branch, image in items:
        provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
        provider.initialize(branch.search_box)
        crop, _, transform = apply_siamfc_cropping(image, tracker.search_size, provider.get(tracker.search_size),
                                                  'bilinear', False, tracker.image_mean)
        crops.append(tracker.normalization(crop / 255.))
        params.append(transform)
    batch = {'z': torch.stack([tracker.anchor for tracker, _, _ in items]),
             'd': torch.stack([branch.template for _, branch, _ in items]), 'x': torch.stack(crops),
             'z_feat_mask': torch.cat([tracker.anchor_mask for tracker, _, _ in items]),
             'd_feat_mask': torch.cat([branch.mask for _, branch, _ in items])}
    with torch.autocast('cuda', dtype=torch.float16):
        candidates = extractor(batch)
        logits = head(candidates).float()
        scores = selection_scores(logits, candidates, extractor.window_penalty)
    choices = scores.masked_fill(~candidates['valid'], -torch.inf).argmax(1).tolist()
    crop_boxes = candidates['boxes'].double().cpu().numpy() * 224.
    observations = []
    for index, ((_, _, image), transform) in enumerate(zip(items, params)):
        boxes = apply_siamfc_cropping_to_boxes(crop_boxes[index], reverse_siamfc_cropping_params(transform))
        for box in boxes:
            bbox_clip_to_image_boundary_(box, np.array((image.shape[-1], image.shape[-2])))
        assert np.isfinite(boxes).all()
        observations.append(({key: value[index:index + 1] for key, value in candidates.items()},
                             logits.sigmoid()[index], boxes, choices[index]))
    return observations


def iou(box, target):
    return float(box_iou(torch.from_numpy(box), torch.from_numpy(target)))


@torch.inference_mode()
def collect_batch(jobs, data, extractor, head, device, args):
    contexts = []
    for sequence_index, query in jobs:
        sequence = data[sequence_index]
        initial = sequence[0].get_bounding_box().copy()
        paths = sequence[0].get_image_path()
        tracker = BoundedRecoveryTracker(extractor, head, read_pair(*map(Path, paths), device), initial, torch.float16)
        # Self-predicted prefix. Later GT is not consulted by tracker.step.
        for frame in range(1, query):
            paths = sequence[frame].get_image_path()
            tracker.step(read_pair(*map(Path, paths), device))
        parent = next(branch for branch in tracker.branches if branch.identifier == tracker.main_id)
        paths = sequence[query].get_image_path()
        image = read_pair(*map(Path, paths), device)
        contexts.append((sequence, query, tracker, parent, image))
    observations = observe_actions([(tracker, parent, image) for _, _, tracker, parent, image in contexts], extractor, head)
    rows, actions = [], []
    for clip_index, ((sequence, query, tracker, parent, image), observation) in enumerate(zip(contexts, observations)):
        candidates, quality, boxes, original_choice = observation
        valid = candidates['valid'][0].cpu().numpy()
        row = {'features': candidates['features'][0].cpu().numpy().copy(),
               'evidence': candidates['evidence'][0].cpu().numpy().copy(),
               'motion': motion_evidence(parent.boxes, boxes, tracker.window),
               'c1_quality': quality.cpu().numpy().copy(), 'valid': valid.copy(),
               'original_choice': original_choice, 'current_iou': np.zeros(len(valid), dtype=np.float32),
               'future_iou': np.zeros((len(valid), args.horizon), dtype=np.float32),
               'wrong_update_fraction': np.zeros(len(valid), dtype=np.float32)}
        rows.append(row)  # Every decision input is materialized before future decode.
        for choice in np.flatnonzero(valid):
            shadow = copy.copy(tracker)
            shadow.stats = tracker.stats.copy()
            shadow.frame = query
            child = shadow._advance(parent, image, boxes, candidates, quality, int(choice), fork=choice != original_choice)
            current_iou = iou(child.box, sequence[query].get_bounding_box())
            row['current_iou'][choice] = current_iou
            wrong_update = int(float(candidates['raw_score'][0, choice]) > .84 and current_iou < .2)
            actions.append({'clip': clip_index, 'choice': int(choice), 'tracker': shadow, 'branch': child,
                            'wrong_updates': wrong_update})
    for offset in range(1, args.horizon + 1):
        images = []
        for sequence, query, _, _, _ in contexts:
            paths = sequence[query + offset].get_image_path()
            images.append(read_pair(*map(Path, paths), device))
        items = [(action['tracker'], action['branch'], images[action['clip']]) for action in actions]
        future = observe_actions(items, extractor, head)
        for action, observation in zip(actions, future):
            shadow = action['tracker']
            shadow.frame += 1
            candidates, quality, boxes, choice = observation
            action['branch'] = shadow._advance(action['branch'], images[action['clip']], boxes, candidates, quality, choice)
            sequence, query, _, _, _ = contexts[action['clip']]
            overlap = iou(action['branch'].box, sequence[query + offset].get_bounding_box())
            rows[action['clip']]['future_iou'][action['choice'], offset - 1] = overlap
            action['wrong_updates'] += int(float(candidates['raw_score'][0, choice]) > .84 and overlap < .2)
    for action in actions:
        rows[action['clip']]['wrong_update_fraction'][action['choice']] = action['wrong_updates'] / (args.horizon + 1)
    for row in rows:
        row['utility'] = row['future_iou'].mean(1) - args.pollution_weight * row['wrong_update_fraction']
        assert all(np.isfinite(row[key]).all() for key in ('features', 'evidence', 'motion', 'c1_quality', 'utility'))
    return rows


def main():
    args = arguments()
    assert args.clips > 0 and args.batch_clips > 0 and 1 <= args.min_history <= args.max_history and args.horizon >= 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    data = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    names = set(split[args.partition])
    eligible = []
    for index in range(len(data)):
        if data[index].get_name() not in names:
            continue
        boxes = data[index].get_all_bounding_boxes()
        valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
        if not valid[0]:
            continue
        queries = [t for t in range(args.min_history, min(args.max_history + 1, len(boxes) - args.horizon))
                   if valid[t:t + args.horizon + 1].all()]
        if queries:
            eligible.append((index, queries))
    assert eligible
    rng = np.random.default_rng(args.seed + (100000 if args.partition == 'validation' else 0))
    jobs = []
    for _ in range(args.clips):
        index, queries = eligible[int(rng.integers(len(eligible)))]
        jobs.append((index, int(rng.choice(queries))))
    checkpoint = torch.load(args.head, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'C1_candidate_quality'
    settings = checkpoint['args']
    extractor = FrozenCandidateExtractor(args.pretrained, settings['candidates'], .45, settings['nms_iou']).to(device)
    head = CandidateQualityHead(settings['hidden']).to(device)
    head.load_state_dict(checkpoint['head'], strict=True)
    head.eval().requires_grad_(False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'source': 'LasHeR TRAIN sequences only, original disjoint C1 split',
                           'jobs': [{'sequence': data[index].get_name(), 'query_frame': t} for index, t in jobs],
                           'eligible_sequences': len(eligible), 'C1_epoch': checkpoint['epoch'],
                           'history': 'C2 predicted prefix; GT only initializes first-frame identity and supplies labels',
                           'future_policy': 'identical frozen C1 continuation for all initial candidate choices',
                           'utility': 'mean next H IoU - lambda * fraction of template writes with IoU<.2 over current+H',
                           'pollution_proxy': 'localization-based, not full distractor identity annotations',
                           'decision_features_saved_before_future_decode': True,
                           'future_action_batch_maximum': args.batch_clips * settings['candidates'],
                           'amp_dtype': 'float16'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    rows, started = [], time.perf_counter()
    for start in range(0, len(jobs), args.batch_clips):
        batch_rows = collect_batch(jobs[start:start + args.batch_clips], data, extractor, head, device, args)
        rows.extend(batch_rows)
        np.savez_compressed(out / f'clips_{start:05d}.npz', **{key: np.stack([row[key] for row in batch_rows]) for key in batch_rows[0]})
        progress = {'completed_clips': len(rows), 'clips': len(jobs), 'elapsed_seconds': time.perf_counter() - started,
                    'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20}
        (out / 'progress.json').write_text(json.dumps(progress, indent=2))
        print('PROGRESS', json.dumps(progress), flush=True)
    np.savez_compressed(out / 'samples.npz', **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
    receipt = {'completed': True, 'clips': len(rows), 'valid_candidate_actions': int(sum(row['valid'].sum() for row in rows)),
               'partition': args.partition, 'elapsed_seconds': time.perf_counter() - started,
               'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'official_tracking_accuracy': False, 'decision_input_contains_future': False}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
