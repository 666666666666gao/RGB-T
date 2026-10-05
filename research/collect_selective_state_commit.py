"""Actual budgeted parent prefixes and matched C actions; GT is offline supervision."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .collect_rollouts import iou
from .collect_temporal import paths
from .evaluate_online import read_pair
from .probe_geometry_commit import private_state, outcome
from .probe_quality_geometry_commit import use_baseline_geometry
from .probe_quality_geometry_rollouts import query_shadow, assert_raw_matches_base
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker
from .selective_state_commit import decision_features, SelectiveStateCommitHead, SelectiveStateCommitTracker
from .temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped

PARENT = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'


def known_box(box):
    return bool(np.isfinite(box).all() and (box[2:] > box[:2]).all())


def family(box, target):
    if not known_box(target):
        return 'unknown'
    overlap = iou(box, target)
    return 'normal' if overlap >= .5 else 'failure' if overlap < .2 else 'uncertain'


def models(args, device):
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    saved = torch.load(args.model, map_location='cpu', weights_only=False)
    assert args.model == PARENT and saved['module'] == 'ABC_recoverability' and saved['epoch'] == 4
    assert saved['args']['c1_head'] == args.c1_head and c1['args']['candidates'] == 5
    extractor = InstanceExtractor(args.pretrained, 5, .45, c1['args']['nms_iou']).to(device)
    modules = RecoverabilityModules(c1).to(device)
    modules.load_state_dict(saved['model'], strict=True)
    modules.eval().requires_grad_(False)
    config = json.loads((Path(args.motion_run) / 'config.json').read_text())
    motion_saved = torch.load(Path(args.motion_run) / 'last.pth', map_location='cpu', weights_only=False)
    assert motion_saved['epoch'] == 30 and config['c1_head'] == args.c1_head
    motion = TemporalModules(c1, config['slots'], config['motion_history'], config['modes'], motion_saved['horizon']).to(device)
    motion.load_state_dict(motion_saved['model'], strict=True)
    motion.eval().requires_grad_(False)
    return extractor, modules, motion, saved['args']['threshold']


@torch.inference_mode()
def query_actions(parent, sequence, job, args):
    q, h = job['query_frame'], job['horizon']
    assert h in (3, args.horizon)
    image = read_pair(*paths(sequence, q), parent.device)
    query = query_shadow(parent, q, 'raw')
    query.step(image)
    d, note = query.query_data, query.query_record
    baseline = torch.tensor([note['baseline_index']], device=parent.device)
    pause = torch.tensor([note['pause']], device=parent.device)
    features, mask = decision_features(parent.modules, d, parent.identity_anchor, parent.identity_memory, baseline, pause)
    proposed = torch.nonzero(mask[0].any(-1)).flatten().tolist()
    proposed.remove(int(baseline[0]))
    indices = [int(baseline[0])] + proposed
    assert 1 <= len(indices) <= 3 and int(d['valid'].sum()) <= 10
    x = np.zeros((3, features.shape[-1]), dtype=np.float32)
    legal = np.zeros((3, 3), dtype=bool)
    ids = np.full(3, -1, dtype=np.int64)
    x[:len(indices)] = features[0, indices].float().cpu().numpy().copy()
    legal[:len(indices)] = mask[0, indices].cpu().numpy().copy()
    ids[:len(indices)] = indices
    keep = int(pause[0])
    assert legal.reshape(-1)[keep]
    # Real NN zero-head check is limited to sanity. It compares output and all
    # parent diagnostic fields/state, ignoring only the new extension counters.
    if args.parity:
        m0 = private_state(parent)
        m0.__class__ = SelectiveStateCommitTracker
        m0.state_commit_head = SelectiveStateCommitHead().to(parent.device).eval().requires_grad_(False)
        m0.stats.update(state_commit_interventions=0, state_geometry_holds=0)
        m0.step(image)
        assert m0.stats.pop('state_commit_interventions') == m0.stats.pop('state_geometry_holds') == 0
        for key in ('state_commit_action', 'committed_search_reference', 'committed_motion_observation'):
            del m0.last_decision[key]
        assert_raw_matches_base(query, m0)
    query.__class__ = RecoverabilityTracker
    actions, shadows, rows = [], [], []
    for action in np.flatnonzero(legal.reshape(-1)):
        slot, mode = divmod(int(action), 3)
        candidate = indices[slot]
        if action == keep:
            shadow = private_state(query)
            write = bool(query.last_decision['template_updated'])
        else:
            shadow = private_state(parent)
            shadow.frame = q
            region, choice = divmod(candidate, 5)
            candidates, quality, boxes, _ = query.current_observations[region]
            _, write, _, _ = RecoverabilityTracker.accept_observation(shadow, image, (candidates, quality, boxes, choice), mode > 0)
            if mode == 2:
                use_baseline_geometry(shadow, query)
            assert shadow.frame == q and shadow.history[-1][1] == q
        actions.append(int(action))
        shadows.append(shadow)
        rows.append({'boxes': [shadow.branch.box.copy()], 'writes': [bool(write)],
                     'search': [shadow.branch.search_box.copy()], 'history': [shadow.history[-1][0].copy()]})
    for offset in range(1, h + 1):
        image = read_pair(*paths(sequence, q + offset), parent.device)
        for shadow, row in zip(shadows, rows):
            row['boxes'].append(shadow.step(image))
            row['writes'].append(bool(shadow.last_decision['template_updated']))
            row['search'].append(shadow.branch.search_box.copy())
            row['history'].append(shadow.history[-1][0].copy())
    # All features and every action/future decision above preceded these labels.
    gt = np.stack([sequence[q + t].get_bounding_box() for t in range(h + 1)])
    known = np.array([known_box(b) for b in gt])
    assert known[0] and known[1:].sum() >= (3 if h == 32 else 1)
    boxes = np.zeros((9, h + 1, 4), dtype=np.float64)
    writes = np.zeros((9, h + 1), dtype=bool)
    overlaps = np.full((9, h + 1), -1., dtype=np.float64)
    search, history = boxes.copy(), boxes.copy()
    outcomes = {}
    for action, row in zip(actions, rows):
        boxes[action], writes[action] = row['boxes'], row['writes']
        search[action], history[action] = row['search'], row['history']
        overlaps[action] = [iou(b, target) if valid else -1. for b, target, valid in zip(row['boxes'], gt, known)]
        outcomes[str(action)] = {str(k): outcome(overlaps[action, :k + 1], writes[action, :k + 1])
                                for k in (3, h) if k <= h}
    baseline_iou = overlaps[keep]
    recovered = any((baseline_iou[t:t + 3] >= .5).all() for t in range(1, h - 1))
    kind = family(boxes[keep, 0], gt[0])
    role = kind if kind != 'failure' else f'recovered_H{h}' if recovered else f'unrecovered_H{h}'
    return query, {'features': x, 'legal': legal, 'candidate_ids': ids, 'keep': np.array(keep),
                   'boxes_xyxy': boxes, 'search_xyxy': search, 'history_xyxy': history,
                   'writes': writes, 'iou': overlaps, 'gt_xyxy': gt, 'gt_known': known}, {
                   **job, 'actual_event_role': role, 'query_family': kind, 'legal_actions': actions,
                   'actual_rollout_horizon': h,
                   'legal_write_pairs': int((legal[:, 0] & legal[:, 1]).sum()),
                   'alternative_candidates': len(indices) - 1, 'baseline_index': int(baseline[0]),
                   'extra_visual_forward_at_query': int(query.last_decision['extra_executed']),
                   'causal_features_before_future': True, 'outcomes': outcomes, 'M0_parity_pass': args.parity}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--model', default=PARENT)
    p.add_argument('--pretrained', default='/data/gb/GOLA/pretrained_models/gola_b224.bin')
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--motion-run', default='/data/gb/outputs/abc_joint_v1_seed42')
    p.add_argument('--jobs-file', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--horizon', type=int, choices=(3, 32), default=32)
    p.add_argument('--parity', action='store_true')
    return p.parse_args()


def main():
    args = arguments()
    jobs = json.loads(Path(args.jobs_file).read_text())['jobs']
    split = json.loads(Path(args.split).read_text())
    assert jobs and not set(split['train']) & set(split['validation'])
    assert all(j['sequence'] in split[j['partition']] for j in jobs)
    assert len({(j['sequence'], j['query_frame']) for j in jobs}) == len(jobs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    extractor, modules, motion, threshold = models(args, device)
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    lookup = {dataset[i].get_name(): i for i in range(len(dataset))}
    (out / 'config.json').write_text(json.dumps(vars(args) | {'features': 917, 'parent_future_policy': 'old4/gross',
        'partition_scope': 'TRAIN-root only; TRAIN and repeated developer validation separated',
        'trainable_scope': 'C extension only; GOLA, C1, old4 A/B/C and motion frozen',
        'max_extra_visual_forwards_per_frame': 1, 'GT_role': 'initialization, eligibility and offline labels'}, indent=2))
    grouped = {}
    for j in jobs:
        grouped.setdefault(j['sequence'], []).append(j)
    started, results = time.perf_counter(), []
    for name, group in grouped.items():
        sequence = dataset[lookup[name]]
        parent = RecoverabilityTracker(extractor, modules, motion, read_pair(*paths(sequence, 0), device),
            sequence[0].get_bounding_box().copy(), threshold, write_verification='action', search_value='gross')
        last_family, event_start = 'normal', 0
        for job in sorted(group, key=lambda j: j['query_frame']):
            q = job['query_frame']
            assert parent.frame < q and q + job['horizon'] < len(sequence)
            while parent.frame + 1 < q:
                frame = parent.frame + 1
                box = parent.step(read_pair(*paths(sequence, frame), device))
                observed_family = family(box, sequence[frame].get_bounding_box())
                if observed_family != last_family:
                    event_start, last_family = frame, observed_family
            parent, arrays, result = query_actions(parent, sequence, job, args)
            if result['query_family'] != last_family:
                event_start, last_family = q, result['query_family']
            result['actual_event_id'] = f'{name}:{last_family}:{event_start}'
            index = len(results)
            np.savez_compressed(out / f'query_{index:05d}.npz', **arrays)
            (out / f'query_{index:05d}.json').write_text(json.dumps(result, indent=2))
            results.append(result)
            print(json.dumps({'completed': len(results), 'total': len(jobs), 'sequence': name, 'query_frame': q,
                              'legal_actions': len(result['legal_actions']), 'elapsed_seconds': time.perf_counter() - started}), flush=True)
    (out / 'summary.json').write_text(json.dumps({'status': 'COMPLETE_MATCHED_STATE_COMMIT_LABELS',
        'queries': len(results), 'sequence_count': len(grouped), 'results': results,
        'distinct_actual_events': len({r['actual_event_id'] for r in results}),
        'legal_write_pair_queries': sum(r['legal_write_pairs'] > 0 for r in results),
        'M0_parity_checked': args.parity, 'elapsed_seconds': time.perf_counter() - started}, indent=2))
    (out / 'COMPLETE').write_text('COMPLETE\n')


if __name__ == '__main__':
    main()
