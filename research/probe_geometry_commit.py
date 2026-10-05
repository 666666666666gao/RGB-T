"""TRAIN query-only causal geometry/appearance commits; current output stays fixed.
C1 keep or past-velocity search prediction is a counterfactual, not verified
identity. Future steps use the same frozen deployed policy, never future GT.
"""
import argparse
import copy
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .collect_rollouts import iou
from .collect_temporal import paths
from .evaluate_online import read_pair
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider

CONTROLS = (('raw', False, False), ('pause', True, False),
            ('geometry_C1', False, True), ('pause_geometry_C1', True, True))


def controls(reference):
    if reference == 'C1':
        return CONTROLS
    if reference == 'C1_components':
        return (('raw', False, False), ('search_C1', False, True),
                ('motion_C1', False, True), ('search_motion_C1', False, True))
    assert reference == 'velocity_search'
    return (('raw', False, False), ('pause', True, False),
            ('search_velocity', False, True), ('pause_search_velocity', True, True))


def next_search_prediction(history, search_box, next_frame):
    """Advance the last two observed centers to the next real frame; retain size."""
    first, last = list(history)[-2:]
    assert last[1] > first[1] and next_frame > last[1]
    c0, c1 = (first[0][:2] + first[0][2:]) / 2, (last[0][:2] + last[0][2:]) / 2
    center = c1 + (c1 - c0) * ((next_frame - last[1]) / (last[1] - first[1]))
    size = search_box[2:] - search_box[:2]
    return np.concatenate((center - size / 2, center + size / 2))


class QueryTracker(RecoverabilityTracker):
    def original_inputs(self, observation, distribution, history):
        self.original_observation = observation
        self.decision_inputs = super().original_inputs(observation, distribution, history)
        return self.decision_inputs


def private_state(parent):
    shadow = copy.copy(parent)
    shadow.branch = parent.branch.copy()
    shadow.branches = [shadow.branch]
    shadow.stats = parent.stats.copy()
    shadow.history = deque(((box.copy(), frame, quality) for box, frame, quality in parent.history), maxlen=8)
    shadow.identity_memory = parent.identity_memory.clone()
    shadow.motion_memory = parent.motion_memory.clone()
    shadow.template_source_box = parent.template_source_box.copy()
    return shadow


def outcome(values, writes):
    known = values >= 0
    longest = current = 0
    for failed in known & (values < .2):
        current = current + 1 if failed else 0
        longest = max(longest, current)
    correct = values >= .5
    first = next((t for t in range(1, len(values) - 2) if correct[t:t + 3].all()), None)
    return {'mean_iou_including_query': float(values[known].mean()) if known.any() else None,
            'mean_future_iou': float(values[1:][known[1:]].mean()) if known[1:].any() else None,
            'valid_GT_frames': int(known.sum()), 'unknown_GT_frames': int((~known).sum()),
            'failure_frames_including_query': int((known & (values < .2)).sum()), 'longest_failure_run': longest,
            'first_future_three_correct_run_offset': first,
            'writes': int(writes.sum()), 'unknown_GT_writes': int((writes & ~known).sum()),
            'wrong_localization_writes': int((writes & known & (values < .2)).sum())}


@torch.inference_mode()
def probe(sequence, job, extractor, modules, motion, device, args, threshold):
    q, horizon = job['query_frame'], max(args.horizons)
    initial = sequence[0].get_bounding_box().copy()
    parent = QueryTracker(extractor, modules, motion, read_pair(*paths(sequence, 0), device), initial,
                          threshold, write_verification=args.write_verification)
    for frame in range(1, q):
        parent.step(read_pair(*paths(sequence, frame), device))
    parent_box, parent_search = parent.branch.box.copy(), parent.branch.search_box.copy()
    parent_history = [(b.copy(), t, quality) for b, t, quality in parent.history]
    parent_identity, parent_motion = parent.identity_memory.clone(), parent.motion_memory.clone()
    image = read_pair(*paths(sequence, q), device)
    query = private_state(parent)
    query_box = query.step(image)
    selected, original = query.last_observation, query.original_observation
    keep_box, keep_quality = original[2][original[3]].copy(), float(original[1][original[3]])
    shadows, rows = [], []
    for name, pause, geometry_changed in controls(args.geometry_reference):
        if args.geometry_reference == 'C1_components':
            pause = bool(query.last_decision['pause'])
        shadow = private_state(parent)
        shadow.frame = q
        branch, write, _, _ = shadow.accept_observation(image, selected, pause)
        assert np.array_equal(branch.box, query_box)
        change_search = geometry_changed and name != 'motion_C1'
        change_motion = geometry_changed and (args.geometry_reference == 'C1' or name in ('motion_C1', 'search_motion_C1'))
        if geometry_changed:
            geometry_c1 = args.geometry_reference in ('C1', 'C1_components')
            requested = keep_box if geometry_c1 else next_search_prediction(parent.history, parent.branch.search_box, q + 1)
            if change_search:
                provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
                provider.initialize(parent.branch.search_box)
                provider.update(float(original[0]['raw_score'][0, original[3]]) if geometry_c1 else 0., requested,
                                np.array((image.shape[-1], image.shape[-2])))
                branch.search_box = provider.cached_bbox.copy()
            if change_motion:
                # Component control changes coordinates only, retaining time and quality.
                observed_quality = keep_quality if args.geometry_reference == 'C1' else shadow.history[-1][2]
                shadow.history[-1] = (keep_box.copy(), q, observed_quality)
        else:
            assert np.array_equal(branch.search_box, query.branch.search_box)
        reference = keep_box if change_motion else query_box
        assert shadow.history[-1][1] == q and np.array_equal(shadow.history[-1][0], reference)
        shadows.append(shadow)
        rows.append({'name': name, 'pause': pause, 'geometry_c1': geometry_changed and args.geometry_reference in ('C1', 'C1_components'),
                     'requested_reference_matched_exactly': not change_search or bool(np.array_equal(branch.search_box, requested)),
                     'boxes': [branch.box.copy()], 'writes': [bool(write)],
                     'search': [branch.search_box.copy()], 'motion': [shadow.history[-1][0].copy()],
                     'extra': [int(query.last_decision['extra_executed'])]})
    if args.geometry_reference == 'C1_components':
        for shadow in shadows:
            assert torch.equal(shadow.identity_memory, shadows[0].identity_memory)
            assert torch.equal(shadow.motion_memory, shadows[0].motion_memory)
            assert torch.equal(shadow.branch.template, shadows[0].branch.template)
            assert [e[1:] for e in shadow.history] == [e[1:] for e in shadows[0].history]
        assert np.array_equal(shadows[2].branch.search_box, shadows[0].branch.search_box)
        assert all(np.array_equal(a[0], b[0]) for a, b in zip(shadows[1].history, shadows[0].history))
    if args.geometry_reference == 'velocity_search':
        for raw_index, predicted_index in [(0, 2), (1, 3)]:
            raw, predicted = shadows[raw_index], shadows[predicted_index]
            assert torch.equal(raw.identity_memory, predicted.identity_memory)
            assert torch.equal(raw.motion_memory, predicted.motion_memory)
            assert torch.equal(raw.branch.template, predicted.branch.template)
            assert all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(raw.history, predicted.history))
    # GT is used only for TRAIN eligibility before launch and offline outcomes below.
    # No GT is passed to query selection, commits, or future policy steps.
    for offset in range(1, horizon + 1):
        image = read_pair(*paths(sequence, q + offset), device)
        for shadow, row in zip(shadows, rows):
            shadow.step(image)
            d = shadow.last_decision
            row['boxes'].append(shadow.branch.box.copy())
            row['writes'].append(bool(d['template_updated']))
            row['search'].append(shadow.branch.search_box.copy())
            row['motion'].append(d['forecast_reference'].double().cpu().numpy().copy())
            row['extra'].append(int(d['extra_executed']))
            assert shadow.frame == q + offset
    assert np.array_equal(parent.branch.box, parent_box) and np.array_equal(parent.branch.search_box, parent_search)
    assert torch.equal(parent.identity_memory, parent_identity) and torch.equal(parent.motion_memory, parent_motion)
    assert len(parent.history) == len(parent_history)
    assert all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(parent.history, parent_history))
    gt = np.stack([sequence[q + t].get_bounding_box() for t in range(horizon + 1)])
    gt_known = np.isfinite(gt).all(1) & (gt[:, 2:] > gt[:, :2]).all(1)
    assert gt_known[1:].sum() >= 3
    results, arrays = [], {'gt_xyxy': gt, 'C1_keep_query_xyxy': keep_box,
                          'gt_known': gt_known,
                          'pre_query_identity_anchor': parent.identity_anchor[0].cpu().numpy(),
                          'pre_query_identity_memory': parent_identity[0].cpu().numpy()}
    arrays.update({'decision_' + key: value[0].detach().cpu().numpy() for key, value in query.decision_inputs.items()})
    for row in rows:
        boxes, writes = np.asarray(row['boxes']), np.asarray(row['writes'], dtype=bool)
        overlap = np.asarray([iou(b, target) if known else -1. for b, target, known in zip(boxes, gt, gt_known)])
        results.append({'name': row['name'], 'query_pause': row['pause'], 'query_geometry_C1': row['geometry_c1'],
                        'query_output_iou': float(overlap[0]) if gt_known[0] else None,
                        'query_geometry_iou': iou(row['search'][0], gt[0]) if gt_known[0] else None,
                        'next_search_reference_iou_at_next_frame': iou(row['search'][0], gt[1]) if gt_known[1] else None,
                        'query_template_updated': bool(writes[0]), 'extra_visual_forwards': sum(row['extra']),
                        'horizons': {str(h): outcome(overlap[:h + 1], writes[:h + 1]) for h in args.horizons}})
        for key, value in [('boxes_xyxy', boxes), ('iou', overlap), ('writes', writes),
                           ('search_xyxy', row['search']), ('motion_xyxy', row['motion'])]:
            arrays[row['name'] + '_' + key] = np.asarray(value)
    search_row = rows[1] if args.geometry_reference == 'C1_components' else rows[2]
    return {'event_id': job['event_id'], 'sequence': sequence.get_name(), 'query_frame': q,
            'selected_matches_C1_keep': bool(np.array_equal(query_box, keep_box)),
            'query_raw_write_eligible': float(selected[0]['raw_score'][0, selected[3]]) > .84,
            'actual_policy_query_pause': bool(query.last_decision['pause']),
            'C1_keep_is_valid_search_reference': (bool(np.array_equal(rows[1]['search'][0], keep_box)) if args.geometry_reference == 'C1_components' else
                                                bool(np.array_equal(rows[2]['search'][0], keep_box)) if args.geometry_reference == 'C1' else None),
            'geometry_reference': args.geometry_reference,
            'query_search_reference_different': bool(not np.array_equal(search_row['search'][0], rows[0]['search'][0])),
            'query_motion_reference_different': bool(not np.array_equal(rows[2]['motion'][0], rows[0]['motion'][0])),
            'query_requested_reference_matched_exactly': search_row['requested_reference_matched_exactly'],
            'query_search_reference_center_shift_pixels': float(np.linalg.norm((search_row['search'][0][:2] + search_row['search'][0][2:] - rows[0]['search'][0][:2] - rows[0]['search'][0][2:]) / 2)),
            'query_visual_work_shared_once': True, 'controls': results}, arrays


def mean_known(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def event_summary(rows, horizons):
    names = [c['name'] for c in rows[0]['controls']]
    groups = {}
    for row in rows:
        groups.setdefault((row['sequence'], row['event_id']), []).append(row)
    events = []
    for (sequence, event_id), states in groups.items():
        averages = {}
        for name in names:
            averages[name] = {str(h): {
                metric: mean_known([next(c for c in row['controls'] if c['name'] == name)['horizons'][str(h)][metric]
                                      for row in states])
                for metric in ('mean_iou_including_query', 'mean_future_iou', 'failure_frames_including_query',
                               'longest_failure_run', 'writes', 'wrong_localization_writes',
                               'valid_GT_frames', 'unknown_GT_frames', 'unknown_GT_writes')}
                for h in horizons}
        events.append({'sequence': sequence, 'event_id': event_id, 'query_states': len(states),
                       'legal_query_write_pairs': sum(row['query_raw_write_eligible'] for row in states),
                       'geometry_position_different_queries': sum(row['query_search_reference_different'] or row['query_motion_reference_different'] for row in states),
                       'search_reference_different_queries': sum(row['query_search_reference_different'] for row in states),
                       'motion_reference_different_queries': sum(row['query_motion_reference_different'] for row in states),
                       'controls_mean_over_correlated_query_states': averages})
    return {'sequences': len({row['sequence'] for row in rows}), 'events': len(events), 'query_states': len(rows),
            'legal_query_write_pairs': sum(row['query_raw_write_eligible'] for row in rows),
            'events_with_legal_write_pairs': sum(e['legal_query_write_pairs'] > 0 for e in events),
            'geometry_position_different_queries': sum(row['query_search_reference_different'] or row['query_motion_reference_different'] for row in rows),
            'events_with_geometry_position_difference': sum(e['geometry_position_different_queries'] > 0 for e in events),
            'requested_reference_not_exact_queries': sum(not row['query_requested_reference_matched_exactly'] for row in rows),
            'event_rows': events,
            'controls_equal_event_mean': {name: {str(h): {
                metric: mean_known([e['controls_mean_over_correlated_query_states'][name][str(h)][metric] for e in events])
                for metric in ('mean_iou_including_query', 'mean_future_iou', 'failure_frames_including_query',
                               'longest_failure_run', 'writes', 'wrong_localization_writes',
                               'valid_GT_frames', 'unknown_GT_frames', 'unknown_GT_writes')}
                for h in horizons} for name in names}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'split', 'jobs-file', 'model', 'pretrained', 'c1-head', 'motion-run', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--horizons', type=int, nargs='+', default=[3, 32])
    p.add_argument('--write-verification', choices=['identity', 'action'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--geometry-reference', choices=['C1', 'velocity_search', 'C1_components'], default='C1')
    args = p.parse_args()
    assert min(args.horizons) > 0 and len(set(args.horizons)) == len(args.horizons)
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    jobs = json.loads(Path(args.jobs_file).read_text())['jobs']
    assert jobs and all(j['sequence'] in split['train'] and j['event_id'] for j in jobs)
    assert len({(j['sequence'], j['query_frame']) for j in jobs}) == len(jobs)
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    indices = {dataset[i].get_name(): i for i in range(len(dataset))}
    assert all(0 < j['query_frame'] and j['query_frame'] + max(args.horizons) < len(dataset[indices[j['sequence']]]) for j in jobs)
    if args.geometry_reference == 'velocity_search':
        assert all(j['query_frame'] >= 2 for j in jobs)
    # Actual TRAIN sampling eligibility; labels never enter probe action inputs.
    for job in jobs:
        sequence = dataset[indices[job['sequence']]]
        eligible_gt = np.stack([sequence[t].get_bounding_box() for t in
                                [0] + list(range(job['query_frame'], job['query_frame'] + max(args.horizons) + 1))])
        known = np.isfinite(eligible_gt).all(1) & (eligible_gt[:, 2:] > eligible_gt[:, :2]).all(1)
        assert known[0] and known[2:].sum() >= 3  # Initialization and at least three future labels; query may be unknown.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    saved = torch.load(args.model, map_location='cpu', weights_only=False)
    assert saved['module'] == 'ABC_recoverability' and saved['args']['c1_head'] == args.c1_head
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
    (out / 'config.json').write_text(json.dumps(vars(args) | {'jobs': jobs, 'checkpoint_epoch': saved['epoch'],
        'scope': 'TRAIN matched-prestate diagnostic; no training or native benchmark',
        'query_output_unchanged': True, 'geometry_reference': args.geometry_reference,
        'geometry_scope': 'C1 changes query search and motion reference; velocity_search changes ONLY next search reference using past centers and real times; C1_components separates query search box and motion-history coordinates with the actual loaded-policy pause, appearance, memories, time and quality held fixed. None is verified identity.',
        'future_policy': 'same frozen checkpoint normal deployed step', 'GT_role': 'initialization, TRAIN prelaunch eligibility, and offline outcomes; no GT action inputs'}, indent=2) + '\n')
    summaries = []
    for index, job in enumerate(jobs):
        result, arrays = probe(dataset[indices[job['sequence']]], job, extractor, modules, motion,
                               device, args, saved['args']['threshold'])
        np.savez_compressed(out / f'event_{index:04d}.npz', **arrays)
        (out / f'event_{index:04d}.json').write_text(json.dumps(result, indent=2) + '\n')
        summaries.append(result)
        print(json.dumps({'completed': index + 1, 'total': len(jobs), 'sequence': job['sequence'],
                          'query_frame': job['query_frame'], 'event_id': job['event_id']}), flush=True)
    (out / 'events.json').write_text(json.dumps({'status': 'COMPLETE_MATCHED_QUERY_COMMIT_ROLLOUTS',
        'queries': len(jobs), 'event_summary': event_summary(summaries, args.horizons), 'results': summaries,
        'aggregation': 'Group by (sequence,event_id), average correlated query offsets within event, then give each event equal weight.',
        'pause_semantics': 'Existing query pause stops template, target identity commits and motion-memory writes; non-target/unknown slots may update.',
        'motion_array_time_semantics': 'Index0 is query history commit; later indices are forecast reference before the corresponding future step.',
        'geometry_reference': args.geometry_reference,
        'search_reference_changed_queries': sum(r['query_search_reference_different'] for r in summaries),
        'motion_reference_changed_queries': sum(r['query_motion_reference_different'] for r in summaries),
        'interpretation': 'Reference substitution only; no claim of trusted geometry or native gain.'}, indent=2) + '\n')
    (out / 'COMPLETE').write_text('COMPLETE\n')


if __name__ == '__main__':
    main()
