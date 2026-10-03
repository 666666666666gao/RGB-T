"""Real TRAIN-prefix witnesses; no optimization or official accuracy claims."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from research import collect_recoverability as collector
from research.recoverability_modules import DECISION_FIELDS, RecoverabilityModules
from research.recoverability_tracker import RecoverabilityTracker
from research.temporal_modules import TemporalModules
from research.train_recoverability import LABEL_FIELDS, load_data
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    out = Path('/data/gb/outputs/recoverability_own_policy_m0_20261003')
    out.mkdir(exist_ok=True)
    split = json.loads(Path('/data/gb/outputs/c1_initial_seed42/split.json').read_text())
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load('/data/wangwj/dataset/LasHeR',
        'trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    indices = {dataset[i].get_name(): i for i in range(len(dataset))}
    c1 = torch.load('/data/gb/outputs/c1_initial_seed42/best.pth', map_location='cpu', weights_only=False)
    checkpoint = torch.load('/data/gb/outputs/recoverability_abc_s42_20261003/best.pth', map_location='cpu', weights_only=False)
    model = RecoverabilityModules(c1).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    zero = RecoverabilityModules(c1).to(device).eval().requires_grad_(False)
    extractor = collector.InstanceExtractor('pretrained_models/gola_b224.bin', 5, .45, c1['args']['nms_iou']).to(device)
    old = torch.load('/data/gb/outputs/abc_joint_v1_seed42/last.pth', map_location='cpu', weights_only=False)
    old_config = json.loads(Path('/data/gb/outputs/abc_joint_v1_seed42/config.json').read_text())
    motion = TemporalModules(c1, old_config['slots'], old_config['motion_history'], old_config['modes'], old['horizon']).to(device)
    motion.load_state_dict(old['model'], strict=True)
    motion.eval().requires_grad_(False)
    args = SimpleNamespace(max_prefix=256, forward_batch=64, prefix_threshold=checkpoint['args']['threshold'],
                           serial_c1_prefix=False)
    real_history, real_step, real_read = collector.history_arrays, RecoverabilityTracker.step, collector.read_pair
    step_records, parents = {}, []

    def tracked_step(tracker, image):
        prior_template, prior_mask = tracker.branch.template.clone(), tracker.branch.mask.clone()
        box = real_step(tracker, image)
        candidates, quality, boxes, choice = tracker.last_observation
        decision = tracker.last_decision
        region = int(decision['region'])
        assert int(decision['choice']) == region * 5 + choice
        assert np.array_equal(box, boxes[choice])
        assert np.array_equal(box, decision['boxes_xyxy'][region, choice].cpu().numpy())
        assert torch.equal(candidates['evidence'][0, choice], decision['evidence'][region, choice])
        assert candidates['raw_score'][0, choice] == decision['raw_score'][region, choice]
        events = [int(decision['extra_executed']),
                  int(region != 0 or choice != int(decision['original_choice'])),
                  int(decision['pause']), int(decision['template_updated'])]
        assert events[3] == int(float(candidates['raw_score'][0, choice]) > .84 and not events[2])
        if events[2]:
            assert float(candidates['raw_score'][0, choice]) > .84
            assert torch.equal(prior_template, tracker.branch.template) and torch.equal(prior_mask, tracker.branch.mask)
        record = {'descriptor': candidates['modality_features'][0, choice].cpu().numpy().copy(),
                  'instance': candidates['instance_features'][0, choice].cpu().numpy().copy(),
                  'evidence': candidates['evidence'][0, choice].cpu().numpy().copy(),
                  'quality': float(quality[choice]), 'box': box.copy(), 'frame': tracker.frame,
                  'write': bool(tracker.last_decision['template_updated'])}
        step_records.setdefault(id(tracker), []).append((record, events))
        return box

    def state(tracker):
        result = {'box': tracker.branch.box.copy(), 'search_box': tracker.branch.search_box.copy(),
                  'template': tracker.branch.template.cpu().numpy().copy(),
                  'mask': tracker.branch.mask.cpu().numpy().copy(),
                  'identity': tracker.identity_memory.cpu().numpy().copy(),
                  'motion': tracker.motion_memory.cpu().numpy().copy(),
                  'history_boxes': np.stack([h[0] for h in tracker.history]),
                  'history_frames': np.asarray([h[1] for h in tracker.history]),
                  'history_quality': np.asarray([h[2] for h in tracker.history]),
                  'frame': np.asarray(tracker.frame), 'pending_frames': np.asarray([h[0] for h in tracker.branch.pending]),
                  'pending_templates': np.asarray([h[1].cpu().numpy() for h in tracker.branch.pending]),
                  'pending_masks': np.asarray([h[2].cpu().numpy() for h in tracker.branch.pending]),
                  'stats': np.asarray([tracker.stats[key] for key in sorted(tracker.stats)]),
                  'anchor': tracker.anchor.cpu().numpy().copy(),
                  'identity_anchor': tracker.identity_anchor.cpu().numpy().copy(),
                  'motion_anchor': tracker.motion_anchor.cpu().numpy().copy()}
        return result

    def checked_history(context, anchor, capacity):
        row = real_history(context, anchor, capacity)
        tracker = context['tracker']
        if isinstance(tracker, RecoverabilityTracker):
            if context['query'] == 1:
                step_records[id(tracker)] = []
            records = step_records[id(tracker)]
            assert len(records) == context['query'] - 1
            history = list(context['history'])
            assert len(history) == context['query']
            for entry, (expected, _) in zip(history[1:], records):
                for key in expected:
                    assert np.array_equal(np.asarray(entry[key]), np.asarray(expected[key])), (key, context['query'])
            count = context['query']
            assert np.array_equal(row['history_valid'], np.arange(capacity) >= capacity - count)
            assert not row['history_write'][:capacity - count].any()
            assert (row['history_frames'][:capacity - count] == 0).all()
            assert np.array_equal(row['history_frames'][-count:], np.arange(count))
            fields = {'descriptor': 'descriptors', 'instance': 'instance_descriptors', 'evidence': 'evidence',
                      'quality': 'quality', 'box': 'boxes', 'frame': 'frames', 'write': 'write'}
            for slot, entry in enumerate(history, capacity - count):
                for source, target in fields.items():
                    key = 'history_' + target
                    expected = anchor if entry[source] is None else entry[source]
                    assert np.array_equal(row[key][slot], np.asarray(expected, dtype=row[key].dtype)), (key, slot)
            assert history[0]['frame'] == 0 and not history[0]['write'] and history[0]['quality'] == 1.
            assert np.array_equal(row['history_boxes'][-count], context['initial'].astype(np.float32))
            parents.append((tracker, state(tracker), row,
                            {key: value.copy() for key, value in row.items()}))
        return row

    RecoverabilityTracker.step, collector.history_arrays = tracked_step, checked_history
    started = time.perf_counter()
    receipt = {'completed': False, 'source': 'real LasHeR TRAIN/held-out GT, existing881/98split',
               'prefix_checkpoint': '/data/gb/outputs/recoverability_abc_s42_20261003/best.pth',
               'prefix_epoch': checkpoint['epoch'], 'official_accuracy': False, 'optimization_updates': 0}

    def collect(named_jobs, prefix, black=False):
        jobs = [(indices[name], query) for name, query in named_jobs]
        for index, query in jobs:
            labels = dataset[index].get_all_bounding_boxes()
            required = labels[[0, query, query + 1, query + 2, query + 3]]
            assert np.isfinite(required).all() and (required[:, 2:] > required[:, :2]).all()
        future_paths = {str(collector.paths(dataset[index], query + offset)[0])
                        for index, query in jobs for offset in range(1, 4)}
        def read(visible, infrared, target_device):
            image = real_read(visible, infrared, target_device)
            return torch.zeros_like(image) if black and str(visible) in future_paths else image
        collector.read_pair = read
        step_records.clear()
        parents.clear()
        rows = collector.collect_batch(jobs, dataset, extractor, model.c1, motion, device, args, prefix)
        for tracker, before, row, decision_snapshot in parents:
            after = state(tracker)
            assert all(np.array_equal(value, after[key]) for key, value in before.items())
            assert all(np.array_equal(value, row[key]) for key, value in decision_snapshot.items())
            recorded = step_records[id(tracker)]
            counts = np.asarray([events for _, events in recorded], dtype=np.int64).sum(0) if recorded else np.zeros(4, dtype=np.int64)
            assert np.array_equal(row['prefix_counts'], counts)
            assert counts[3] == int(row['history_write'].sum())
        for row in rows:
            assert all(np.isfinite(value).all() for value in row.values())
            assert np.array_equal(row['action_valid'][..., 0], row['valid'])
            assert np.array_equal(row['action_valid'][..., 1], row['valid'] & (row['raw_score'] > .84))
        return rows

    default_jobs = [('blkboycoming', 17)]
    assert default_jobs[0][0] in split['train']
    baseline, zero_rows = collect(default_jobs, None), collect(default_jobs, zero)
    shared = DECISION_FIELDS + LABEL_FIELDS
    assert all(np.array_equal(baseline[0][key], zero_rows[0][key]) for key in shared)
    receipt['zero_prefix_c1_all_decision_and_label_arrays_bitwise'] = True
    for partition, named_jobs in [('train', [('elector_1227', 1), ('blkboycoming', 175)]),
                                  ('validation', [('2girlgoleft', 1), ('boybackpack', 209)])]:
        assert all(name in split[partition] for name, _ in named_jobs)
        rows = collect(named_jobs, model)
        if partition == 'validation':
            pause_record, pause_events = step_records[id(parents[1][0])][207]
            assert pause_record['frame'] == 208 and pause_events[1] and pause_events[2] and not pause_events[3]
            assert int(parents[1][0].last_decision['region']) > 0
            assert rows[1]['prefix_counts'][0] > 0 and rows[1]['prefix_counts'][1] > 0 and rows[1]['prefix_counts'][2] > 0
            receipt['real_extra_candidate_selection_and_high_confidence_pause208_verified'] = True
        black_rows = collect(named_jobs, model, True)
        assert all(np.array_equal(row[key], other[key]) for row, other in zip(rows, black_rows) for key in DECISION_FIELDS)
        assert any(not np.array_equal(row['future_iou'], other['future_iou']) for row, other in zip(rows, black_rows))
        receipt[partition] = {'jobs': named_jobs, 'actual_prefix_counts_sum': np.stack([r['prefix_counts'] for r in rows]).sum(0).tolist(),
                              'all_recorded_prefix_steps_exact': True, 'parent_state_unchanged_after_future': True,
                              'future_image_perturbation_preserves_decision_arrays': True, 'future_labels_changed': True}
    for name, rows, policy in [('c1', baseline, 'frozen full GOLA/C1'), ('zero', zero_rows, 'frozen learned ABC')]:
        directory = out / name
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / 'samples.npz', **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
        (directory / 'config.json').write_text(json.dumps({'partition': 'train', 'clips': 1,
            'prefix_policy': policy, 'jobs': [{'sequence': 'blkboycoming', 'query_frame': 17}]}))
        (directory / 'completion.json').write_text(json.dumps({'partition': 'train', 'completed': True,
            'decision_input_contains_future': False, 'clips': 1}))
    mixed, _, _, mixed_jobs = load_data([out / 'c1', out / 'c1', out / 'zero'], 'train', device)
    assert len(mixed_jobs) == len(mixed['valid']) == 2
    receipt.update(completed=True, mixed_cache_preserves_two_policies_and_deduplicates_same_policy=True,
                   elapsed_seconds=time.perf_counter() - started, peak_cuda_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                   prefix_counts_columns=['extra_visual_forwards', 'changed_candidate_indices', 'paused_query_writes', 'template_updates'])
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
