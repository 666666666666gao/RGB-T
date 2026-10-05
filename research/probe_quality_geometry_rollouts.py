"""Matched query consequences on LasHeR TRAIN-root development events, not native scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .collect_rollouts import iou
from .collect_temporal import paths
from .evaluate_online import read_pair
from .probe_geometry_commit import private_state, outcome, mean_known
from .probe_quality_geometry_commit import QualityGeometryQueryTracker, MODES
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def query_shadow(parent, query, mode):
    shadow = private_state(parent)
    shadow.__class__ = QualityGeometryQueryTracker
    shadow.query, shadow.probe_mode, shadow.query_record = query, mode, None
    return shadow


def assert_raw_matches_base(raw, base):
    assert raw.frame == base.frame and raw.stats == base.stats
    for field in ('box', 'search_box', 'template', 'mask'):
        a, b = getattr(raw.branch, field), getattr(base.branch, field)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else np.array_equal(a, b)
    assert torch.equal(raw.identity_memory, base.identity_memory)
    assert torch.equal(raw.motion_memory, base.motion_memory)
    assert all(np.array_equal(a[0], b[0]) and a[1:] == b[1:]
               for a, b in zip(raw.history, base.history))
    assert len(raw.history) == len(base.history)
    assert raw.last_decision.keys() == base.last_decision.keys()
    assert all(torch.equal(raw.last_decision[k], base.last_decision[k]) for k in raw.last_decision)


@torch.inference_mode()
def probe(sequence, job, extractor, modules, motion, args, device, threshold):
    query, horizon = job['query_frame'], max(args.horizons)
    parent = RecoverabilityTracker(extractor, modules, motion, read_pair(*paths(sequence, 0), device),
                                  sequence[0].get_bounding_box().copy(), threshold,
                                  write_verification='action', search_value='gross')
    for frame in range(1, query):
        parent.step(read_pair(*paths(sequence, frame), device))
    shadows = [query_shadow(parent, query, mode) for mode in MODES]
    baseline = private_state(parent)
    prefix_box, prefix_search = parent.branch.box.copy(), parent.branch.search_box.copy()
    prefix_identity, prefix_motion = parent.identity_memory.clone(), parent.motion_memory.clone()
    rows = [{key: [] for key in ('boxes', 'writes', 'search', 'motion', 'extra')} for _ in MODES]
    for offset in range(horizon + 1):
        image = read_pair(*paths(sequence, query + offset), device)
        baseline.step(image)
        for shadow, row in zip(shadows, rows):
            box = shadow.step(image)
            row['boxes'].append(box.copy())
            row['writes'].append(bool(shadow.last_decision['template_updated']))
            row['search'].append(shadow.branch.search_box.copy())
            row['motion'].append(shadow.history[-1][0].copy())
            row['extra'].append(int(shadow.last_decision['extra_executed']))
            assert shadow.frame == query + offset
        assert_raw_matches_base(shadows[0], baseline)
        if offset == 0:
            q = [s.query_record for s in shadows]
            assert q[0]['chosen_index'] == job['saved_baseline_index']
            assert q[1]['chosen_index'] == job['saved_quality_index']
            assert len({r['chosen_index'] for r in q[1:]}) == 1
            assert all(np.array_equal(q[1]['output_box'], r['output_box']) for r in q[2:])
            assert torch.equal(shadows[2].branch.template, shadows[3].branch.template)
            assert torch.equal(shadows[2].identity_memory, shadows[3].identity_memory)
            assert torch.equal(shadows[2].motion_memory, shadows[3].motion_memory)
            assert np.array_equal(shadows[3].branch.search_box, baseline.branch.search_box)
            assert np.array_equal(shadows[3].history[-1][0], baseline.history[-1][0])
            assert shadows[3].history[-1][1:] == baseline.history[-1][1:]
    assert parent.frame == query - 1
    assert np.array_equal(parent.branch.box, prefix_box) and np.array_equal(parent.branch.search_box, prefix_search)
    assert torch.equal(parent.identity_memory, prefix_identity) and torch.equal(parent.motion_memory, prefix_motion)
    # Offline labels only: all query choices and future frames have already run.
    gt = np.stack([sequence[query + t].get_bounding_box() for t in range(horizon + 1)])
    known = np.isfinite(gt).all(1) & (gt[:, 2:] > gt[:, :2]).all(1)
    if horizon == 32:
        assert known[1:].sum() >= 3
    arrays, controls = {'gt_xyxy': gt, 'gt_known': known}, []
    for mode, shadow, row in zip(MODES, shadows, rows):
        boxes, writes = np.asarray(row['boxes']), np.asarray(row['writes'], dtype=bool)
        overlap = np.array([iou(b, target) if valid else -1. for b, target, valid in zip(boxes, gt, known)])
        controls.append({'mode': mode, 'chosen_index': shadow.query_record['chosen_index'],
                         'legal_query_write_pair': shadow.query_record['legal_write_pair'],
                         'query_write': bool(writes[0]), 'query_pause': shadow.query_record['pause'],
                         'query_iou': float(overlap[0]) if known[0] else None,
                         'extra_visual_forwards': sum(row['extra']),
                         'horizons': {str(h): outcome(overlap[:h + 1], writes[:h + 1]) for h in args.horizons}})
        for key, value in [('boxes_xyxy', boxes), ('iou', overlap), ('writes', writes),
                           ('search_xyxy', row['search']), ('history_xyxy', row['motion'])]:
            arrays[mode + '_' + key] = np.asarray(value)
    return {'sequence': job['sequence'], 'event_id': job['event_id'], 'event_kind': job['event_kind'],
            'query_frame': query, 'raw_base_exact_all_frames': True,
            'same_prequery_state': True, 'controls': controls,
            'quality_changed_query': controls[0]['chosen_index'] != controls[1]['chosen_index'],
            'geometry_changed_query': not np.array_equal(rows[2]['search'][0], rows[3]['search'][0])
                                      or not np.array_equal(rows[2]['motion'][0], rows[3]['motion'][0])}, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'split', 'jobs-file', 'model', 'pretrained', 'c1-head', 'motion-run', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 32])
    args = parser.parse_args()
    # The existing memory-mapped TRAIN cache stores paths relative to LasHeR,
    # including traingset/. The online directory evaluator uses a different root.
    assert args.root == '/data/wangwj/dataset/LasHeR'
    assert args.model == '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
    assert min(args.horizons) > 0 and len(set(args.horizons)) == len(args.horizons)
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    jobs = json.loads(Path(args.jobs_file).read_text())['jobs']
    assert jobs and all(j['sequence'] in split['validation'] for j in jobs)
    assert len({(j['sequence'], j['event_id']) for j in jobs}) == len(jobs)
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    indices = {dataset[i].get_name(): i for i in range(len(dataset))}
    assert all(0 < j['query_frame'] and j['query_frame'] + max(args.horizons) < len(dataset[indices[j['sequence']]]) for j in jobs)
    assert all(len(dataset[indices[j['sequence']]]) == j['total_frames'] for j in jobs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    saved = torch.load(args.model, map_location='cpu', weights_only=False)
    assert saved['module'] == 'ABC_recoverability' and saved['epoch'] == 4
    assert saved['args']['c1_head'] == args.c1_head
    extractor = InstanceExtractor(args.pretrained, c1['args']['candidates'], .45, c1['args']['nms_iou']).to(device)
    modules = RecoverabilityModules(c1).to(device)
    modules.load_state_dict(saved['model'], strict=True)
    modules.eval().requires_grad_(False)
    config = json.loads((Path(args.motion_run) / 'config.json').read_text())
    motion_saved = torch.load(Path(args.motion_run) / 'last.pth', map_location='cpu', weights_only=False)
    assert motion_saved['epoch'] == 30 and config['c1_head'] == args.c1_head
    motion = TemporalModules(c1, config['slots'], config['motion_history'], config['modes'], motion_saved['horizon']).to(device)
    motion.load_state_dict(motion_saved['model'], strict=True)
    motion.eval().requires_grad_(False)
    (out / 'config.json').write_text(json.dumps(vars(args) | {
        'scope': 'Repeated-development internal validation subset of TRAIN-root; diagnostic, not native accuracy.',
        'jobs': jobs, 'new_training_steps': 0, 'future_policy': 'same frozen parent',
        'GT_role': 'First-frame initialization and offline event/outcome labels only.'}, indent=2))
    results = []
    for index, job in enumerate(jobs):
        result, arrays = probe(dataset[indices[job['sequence']]], job, extractor, modules, motion,
                               args, device, saved['args']['threshold'])
        np.savez_compressed(out / f'event_{index:04d}.npz', **arrays)
        (out / f'event_{index:04d}.json').write_text(json.dumps(result, indent=2))
        results.append(result)
        print(json.dumps({'completed': index + 1, 'total': len(jobs), **job}), flush=True)
    averages = {mode: {str(h): {metric: mean_known([
        next(c for c in r['controls'] if c['mode'] == mode)['horizons'][str(h)][metric] for r in results])
        for metric in ('mean_future_iou', 'longest_failure_run', 'wrong_localization_writes')}
        for h in args.horizons} for mode in MODES}
    (out / 'events.json').write_text(json.dumps({'status': 'COMPLETE_MATCHED_QUERY_CONTROLS',
        'event_count': len(results), 'results': results, 'controls_equal_event_mean': averages,
        'raw_base_exact_all_frames': True, 'new_training_steps': 0,
        'limitation': 'One query per sampled event; repeated developer videos, not independent native confirmation.'}, indent=2))
    (out / 'COMPLETE').write_text('COMPLETE\n')


if __name__ == '__main__':
    main()
