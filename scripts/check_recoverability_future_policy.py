"""Real-video state/causality witness for a frozen own-policy future teacher."""
import argparse
import copy
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from research.recoverability_modules import DECISION_FIELDS, RecoverabilityModules
from research.temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(tracker, stats=True):
    branch = tracker.branch
    result = {'frame': tracker.frame, 'box': branch.box.copy(), 'search_box': branch.search_box.copy(),
              'template': branch.template.clone(), 'mask': branch.mask.clone(),
              'committed_template': branch.committed_template.clone(), 'committed_mask': branch.committed_mask.clone(),
              'pending': [(f, z.clone(), m.clone()) for f, z, m in branch.pending],
              'scores': list(branch.scores), 'boxes': [b.copy() for b in branch.boxes],
              'identity': tracker.identity_memory.clone(), 'motion': tracker.motion_memory.clone(),
              'history': [(b.copy(), f, q) for b, f, q in tracker.history],
              'source_frame': tracker.template_source_frame, 'source_box': tracker.template_source_box.copy()}
    if stats:
        result['stats'] = tracker.stats.copy()
    return result


def equal(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal(a, b)
    elif isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected)
    else:
        assert actual == expected


def fork(tracker):
    child = copy.copy(tracker)
    child.stats = tracker.stats.copy()
    child.history = deque(tracker.history, maxlen=8)
    return child


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', required=True)
    parser.add_argument('--old-tracker', required=True)
    parser.add_argument('--old-collector', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    candidate_dir = Path(args.candidate_dir)
    old_tracker = load_source('research._future_before_tracker', args.old_tracker).RecoverabilityTracker
    tracker_module = load_source('research._future_candidate_tracker', candidate_dir / 'recoverability_tracker.py')
    tracker_type = tracker_module.RecoverabilityTracker
    collector = load_source('research._future_candidate_collector', candidate_dir / 'collect_recoverability.py')
    old_collector = load_source('research._future_before_collector', args.old_collector)
    import research.recoverability_tracker as production_tracker

    c1 = torch.load('/data/gb/outputs/c1_initial_seed42/best.pth', map_location='cpu', weights_only=False)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 4
    model = RecoverabilityModules(c1).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    config = json.loads(Path('/data/gb/outputs/abc_joint_v1_seed42/config.json').read_text())
    old = torch.load('/data/gb/outputs/abc_joint_v1_seed42/last.pth', map_location='cpu', weights_only=False)
    motion = TemporalModules(c1, config['slots'], config['motion_history'], config['modes'], old['horizon']).to(device)
    motion.load_state_dict(old['model'], strict=True)
    motion.eval().requires_grad_(False)
    extractor = collector.InstanceExtractor('pretrained_models/gola_b224.bin', 5, .45, c1['args']['nms_iou']).to(device)
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load('/data/wangwj/dataset/LasHeR',
        'trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    indices = {dataset[i].get_name(): i for i in range(len(dataset))}
    started = time.perf_counter()
    parity = []
    helper_checks = 0
    for verification in ('action', 'identity'):
        for name, count, expect_write in (('3whitemen', 449, False), ('ab_motocometurn', 172, True)):
            sequence = dataset[indices[name]]
            image = collector.read_pair(*collector.paths(sequence, 0), device)
            init = sequence[0].get_bounding_box().copy()
            before = old_tracker(extractor, model, motion, image, init, checkpoint['args']['threshold'],
                                 write_verification=verification)
            current = tracker_type(extractor, model, motion, image, init, checkpoint['args']['threshold'],
                                   write_verification=verification)
            checked_helper = False
            for frame in range(1, count):
                image = collector.read_pair(*collector.paths(sequence, frame), device)
                parent = fork(current)
                parent_state = snapshot(parent)
                equal(before.step(image), current.step(image))
                equal(snapshot(before), snapshot(current))
                equal(before.last_decision, current.last_decision)
                equal(snapshot(parent), parent_state)
                observation = current.last_observation
                selected_write = float(observation[0]['raw_score'][0, observation[3]]) > .84
                if not checked_helper and frame >= 8 and selected_write:
                    accepted = []
                    for pause in (False, True):
                        child = fork(parent)
                        child.frame = frame
                        branch, wrote, rates, verified = child.accept_observation(image, observation, pause)
                        assert wrote == (not pause)
                        assert np.array_equal(branch.box, observation[2][observation[3]])
                        assert child.history[-1][1] == frame and len(child.history) <= 8
                        assert all(frame - entry[0] < child.window for entry in branch.pending)
                        assert torch.equal(rates[0, :2].sum(0) > 0, verified[0] & wrote)
                        if pause:
                            assert torch.equal(branch.template, parent.branch.template)
                            assert not rates[0, :2].any()
                        if pause == bool(current.last_decision['pause']):
                            equal(snapshot(child, stats=False), snapshot(current, stats=False))
                        accepted.append(child)
                        equal(snapshot(parent), parent_state)
                    # Interleaved future steps cannot mutate either sibling or its parent.
                    for offset in range(1, 4):
                        future = collector.read_pair(*collector.paths(sequence, frame + offset), device)
                        untouched = snapshot(accepted[1])
                        accepted[0].step(future)
                        equal(snapshot(accepted[1]), untouched)
                        untouched = snapshot(accepted[0])
                        accepted[1].step(future)
                        equal(snapshot(accepted[0]), untouched)
                        equal(snapshot(parent), parent_state)
                    helper_checks += 1
                    checked_helper = True
            assert checked_helper == expect_write
            parity.append({'verification': verification, 'sequence': name, 'frames': count,
                           'stats': current.stats, 'all_boxes_states_timelines_bitwise_equal': True})
            print('PARITY', json.dumps(parity[-1]), flush=True)
    assert sum(x['stats']['extra_visual_forwards'] for x in parity) > 0
    assert sum(x['stats']['paused_query_writes'] for x in parity) > 0
    assert sum(x['stats']['changed_candidate_indices'] for x in parity) > 0

    training = json.loads(Path('/data/gb/outputs/recoverability_train_s42_20261003/config.json').read_text())
    jobs = []
    for job in training['jobs']:
        index = indices[job['sequence']]
        if index in [i for i, _ in jobs]:
            continue
        gt = dataset[index].get_all_bounding_boxes()[:21]
        if np.isfinite(gt).all() and (gt[:, 2:] > gt[:, :2]).all():
            jobs.append((index, 1 if not jobs else 17))
        if len(jobs) == 2:
            break
    assert len(jobs) == 2
    # The two boundary queries have no writable candidate; this cached TRAIN
    # query has raw>.84 and exercises forced regular/pause label generation.
    jobs.append((indices['carstart189'], 9))
    options = SimpleNamespace(max_prefix=64, forward_batch=1,
                              prefix_threshold=checkpoint['args']['threshold'], serial_c1_prefix=False)
    production_tracker.RecoverabilityTracker = old_tracker
    baseline = old_collector.collect_batch(jobs, dataset, extractor, model.c1, motion, device, options, model)
    production_tracker.RecoverabilityTracker = tracker_type
    default = collector.collect_batch(jobs, dataset, extractor, model.c1, motion, device, options, model)
    for a, b in zip(default, baseline):
        equal(a, b)
    c1_rows = collector.collect_batch(jobs, dataset, extractor, model.c1, motion, device, options, model,
                                      future_policy='c1', prefix_write_verification='action')
    parents, saved_inputs, future_calls, actions = [], [], [], {}
    next_action = [0] * len(jobs)
    real_arrays, real_paths = collector.history_arrays, collector.paths
    real_step, real_accept = tracker_type.step, tracker_type.accept_observation

    def capture_arrays(context, anchor, capacity):
        row = real_arrays(context, anchor, capacity)
        parents.append((context['tracker'], snapshot(context['tracker']), row, context['query']))
        return row

    def capture_paths(sequence, frame):
        queries = {dataset[i].get_name(): q for i, q in jobs}
        if frame > queries[sequence.get_name()] and not saved_inputs:
            assert len(parents) == len(jobs)
            saved_inputs.extend({k: row[k].copy() for k in DECISION_FIELDS} for _, _, row, _ in parents)
        return real_paths(sequence, frame)

    def capture_step(tracker, image):
        audited = len(parents) == len(jobs) and all(tracker is not p[0] for p in parents)
        old_frame = tracker.frame
        result = real_step(tracker, image)
        if audited:
            assert all(tracker.history is not p[0].history and tracker.stats is not p[0].stats for p in parents)
            assert tracker.frame == old_frame + 1
            action = actions[id(tracker)]
            overlap = collector.iou(result, dataset[jobs[action['clip']][0]][tracker.frame].get_bounding_box())
            record = {'frame': tracker.frame, 'iou': overlap,
                      'write': bool(tracker.last_decision['template_updated'])}
            action['future'].append(record)
            future_calls.append(record)
        return result

    def capture_accept(tracker, image, observation, pause):
        result = real_accept(tracker, image, observation, pause)
        if len(parents) == len(jobs) and all(tracker is not p[0] for p in parents):
            clip = next(i for i, p in enumerate(parents) if tracker.identity_anchor is p[0].identity_anchor)
            query = jobs[clip][1]
            if tracker.frame == query:
                row = parents[clip][2]
                keys = [(r, int(k), p) for r in range(7) for k in np.flatnonzero(row['valid'][r])
                        for p in range(2 if float(row['raw_score'][r, k]) > .84 else 1)]
                region, choice, expected_pause = keys[next_action[clip]]
                next_action[clip] += 1
                assert observation[3] == choice and pause == bool(expected_pause)
                assert np.array_equal(observation[2].astype(np.float32), row['image_boxes'][region])
                overlap = collector.iou(result[0].box, dataset[jobs[clip][0]][query].get_bounding_box())
                actions[id(tracker)] = {'clip': clip, 'region': region, 'choice': choice, 'pause': expected_pause,
                                        'query_wrong_write': int(result[1] and overlap < .2), 'future': []}
        return result

    collector.history_arrays, collector.paths = capture_arrays, capture_paths
    tracker_type.step, tracker_type.accept_observation = capture_step, capture_accept
    own_rows = collector.collect_batch(jobs, dataset, extractor, model.c1, motion, device, options, model,
                                       future_policy='own', prefix_write_verification='action')
    collector.history_arrays, collector.paths = real_arrays, real_paths
    tracker_type.step, tracker_type.accept_observation = real_step, real_accept
    expected_steps = sum(int(row['action_valid'].sum()) for row in own_rows) * 3
    assert len(future_calls) == expected_steps
    assert len(actions) * 3 == expected_steps
    for action in actions.values():
        clip, region, choice, pause = (action[k] for k in ('clip', 'region', 'choice', 'pause'))
        assert [x['frame'] for x in action['future']] == [jobs[clip][1] + n for n in (1, 2, 3)]
        assert np.array_equal(np.asarray([x['iou'] for x in action['future']], dtype=np.float32),
                              own_rows[clip]['future_iou'][region, choice, pause])
        wrong = action['query_wrong_write'] + sum(int(x['write'] and x['iou'] < .2) for x in action['future'])
        assert own_rows[clip]['wrong_update_fraction'][region, choice, pause] == wrong / 4
    assert len(saved_inputs) == len(jobs)
    for index, ((parent, original, _, query), own, reference) in enumerate(zip(parents, own_rows, c1_rows)):
        equal(snapshot(parent), original)
        assert parent.frame == query - 1
        for key in DECISION_FIELDS:
            equal(own[key], reference[key])
            equal(own[key], saved_inputs[index][key])
        for key in ('current_iou', 'history_iou', 'motion_targets', 'action_valid'):
            equal(own[key], reference[key])
        assert (own['valid'] & (own['raw_score'] <= .84)).any()
        assert own['valid'][1:].any()
    assert any(row['action_valid'][..., 1].any() for row in own_rows)
    reverse = collector.collect_batch(list(reversed(jobs)), dataset, extractor, model.c1, motion, device, options, model,
                                      future_policy='own', prefix_write_verification='action')
    for a, b in zip(own_rows, reversed(reverse)):
        equal(a, b)
    assert all(p.grad is None for module in (model, motion, extractor) for p in module.parameters())
    result = {'status': 'PASS', 'model': args.model, 'epoch': checkpoint['epoch'],
              'old_new_full_video_parity': parity, 'query_helper_regular_pause_and_sibling_checks': helper_checks,
              'default_c1_collector_all_arrays_bitwise_equal': True,
              'own_c1_query_inputs_and_current_labels_bitwise_equal': True,
              'inputs_copied_before_future_decode': True, 'parents_unchanged_after_all_own_rollouts': True,
              'query_frames': [q for _, q in jobs], 'own_future_step_calls': len(future_calls),
              'expected_own_future_step_calls': expected_steps, 'actual_future_frames_and_gt_labels_equal': True,
              'actual_query_and_future_writes_recompute_all_wrong_update_labels': True,
              'reversed_job_order_labels_bitwise_equal': True,
              'future_label_max_difference': max(float(np.abs(a['future_iou']-b['future_iou']).max()) for a, b in zip(own_rows, c1_rows)),
              'wrong_write_label_max_difference': max(float(np.abs(a['wrong_update_fraction']-b['wrong_update_fraction']).max()) for a, b in zip(own_rows, c1_rows)),
              'no_optimizer': True, 'official_accuracy': False,
              'elapsed_seconds': time.perf_counter()-started, 'peak_cuda_mib': torch.cuda.max_memory_allocated()/2**20}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for label, rows in (('c1', c1_rows), ('own', own_rows)):
        np.savez_compressed(out/(label+'.npz'), **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
    (out/'receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
