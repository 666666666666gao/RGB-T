"""Audit completed bestold4 own-policy shards against TRAIN GT, then concatenate exactly."""
import argparse
import ast
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--partition', choices=('train', 'validation'), required=True)
args = p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and Path.cwd() == Path('/data/gb/GOLA')
setup, outputs = Path('/data/gb/setup'), Path('/data/gb/outputs')
out = outputs / 'recoverability_best_policy_merged_20261004/own' / args.partition
assert not out.exists()
started = time.perf_counter()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
review = read(setup / 'best_policy_merge_source_review_20261004.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
probe = read(setup / 'best_policy_probe_actual_cpu_acceptance_20261004.json')
assert probe['status'] == 'PASS' and probe['both_partitions_passed'] and probe['clips_per_partition'] == 16
assert probe['prefix_epoch'] == 4 and probe['future_teacher_is_global_best']
parent = outputs / 'recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
assert sha(parent) == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
reference = read(outputs / 'recoverability_current_policy_merged_20261004/own' / args.partition / 'config.json')
expected_jobs = reference['jobs']
expected_count = {'train': 902, 'validation': 128}[args.partition]
assert len(expected_jobs) == len({(job['sequence'], job['query_frame']) for job in expected_jobs}) == expected_count
split = read(reference['split'])
assert len(split['train']) == 881 and len(split['validation']) == 98 and not set(split['train']) & set(split['validation'])
assert {job['sequence'] for job in expected_jobs} <= set(split[args.partition])
dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(reference['root'], reference['cache'])
sequences = {dataset[index].get_name(): dataset[index] for index in range(len(dataset))}
tree = ast.parse(Path('research/recoverability_modules.py').read_text())
decision_fields = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                      and any(isinstance(t, ast.Name) and t.id == 'DECISION_FIELDS' for t in n.targets)))
assert not set(decision_fields) & {'current_iou', 'future_iou', 'history_iou', 'wrong_update_fraction', 'action_valid'}
shards = [('train_probe', 16), ('train0_rest', 296), ('train1_rest', 295), ('train2_rest', 295)] if args.partition == 'train' else [('validation_probe', 16), ('validation_rest', 112)]
common = ('prefix_model', 'prefix_checkpoint_epoch', 'prefix_policy', 'prefix_write_verification', 'prefix_execution',
          'future_policy', 'future_policy_mode', 'future_checkpoint_epoch', 'future_execution', 'future_horizon',
          'root', 'cache', 'head', 'pretrained', 'motion_run', 'split', 'regions', 'max_prefix', 'batch_clips', 'forward_batch', 'seed')
arrays, configs, receipts, evidence, jobs = [], [], [], [], []

def overlaps(boxes, gt):
    intersection = np.maximum(np.minimum(boxes[..., 2:], gt[..., 2:]) - np.maximum(boxes[..., :2], gt[..., :2]), 0).prod(-1)
    areas = np.maximum(boxes[..., 2:] - boxes[..., :2], 0).prod(-1)
    return intersection / (areas + (gt[..., 2:] - gt[..., :2]).prod(-1) - intersection)

for shard, count in shards:
    root = outputs / ('recoverability_best_policy_collect_' + shard + '_20261004')
    assert (root / 'job_completed.txt').is_file(), str(root)
    cfg, done = read(root / 'config.json'), read(root / 'completion.json')
    assert done['completed'] and not done['decision_input_contains_future'] and not done['official_tracking_accuracy']
    assert cfg['partition'] == done['partition'] == args.partition and cfg['clips'] == done['clips'] == len(cfg['jobs']) == count
    assert cfg['prefix_model'] == str(parent) and cfg['prefix_checkpoint_epoch'] == cfg['future_checkpoint_epoch'] == 4
    assert cfg['future_policy_mode'] == 'own' and cfg['prefix_write_verification'] == 'action'
    assert cfg['prefix_execution'] == 'per-sequence' and cfg['future_execution'] == 'per-action serial, identical deployed step and write verification'
    assert cfg['decision_inputs_copied_before_future_decode'] and cfg['future_horizon'] == 3
    assert cfg['max_prefix'] == 1024 and cfg['batch_clips'] == 16 and cfg['forward_batch'] == 64 and cfg['seed'] == 42
    for key in ('root', 'cache', 'head', 'pretrained', 'motion_run', 'split', 'regions'):
        assert cfg[key] == reference[key], key
    with np.load(root / 'samples.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert set(decision_fields) <= set(data) and all(len(value) == count and np.isfinite(value).all() for value in data.values())
    assert data['instance_features'].shape == (count, 7, 5, 2, 768)
    assert data['image_boxes'].shape == (count, 7, 5, 4) and data['current_iou'].shape == data['valid'].shape == (count, 7, 5)
    assert data['action_valid'].shape == (count, 7, 5, 2) and data['future_iou'].shape == (count, 7, 5, 2, 3)
    assert data['history_boxes'].shape == (count, 1024, 4) and data['history_frames'].shape == data['history_valid'].shape == data['history_iou'].shape == (count, 1024)
    assert np.array_equal(data['action_valid'][..., 0], data['valid'])
    assert np.array_equal(data['action_valid'][..., 1], data['valid'] & (data['raw_score'] > .84))
    assert data['valid'][np.arange(count), 0, data['original_choice']].all()
    assert int(data['action_valid'].sum()) == done['valid_actions']
    assert (data['current_iou'][data['valid']] >= 0).all() and (data['current_iou'][data['valid']] <= 1).all()
    assert (data['future_iou'][data['action_valid']] >= 0).all() and (data['future_iou'][data['action_valid']] <= 1).all()
    assert (data['wrong_update_fraction'][data['action_valid']] >= 0).all() and (data['wrong_update_fraction'][data['action_valid']] <= 1).all()
    current_errors, history_errors = [], []
    for index, job in enumerate(cfg['jobs']):
        sequence, frame = sequences[job['sequence']], job['query_frame']
        past = data['history_valid'][index].astype(bool)
        frames = data['history_frames'][index][past]
        assert np.array_equal(frames, np.arange(max(0, frame - 1024), frame)), (shard, job)
        target = sequence[frame].get_bounding_box()
        assert np.isfinite(target).all() and (target[2:] > target[:2]).all()
        measured = overlaps(data['image_boxes'][index].astype(np.float64), target)
        error = np.abs(measured - data['current_iou'][index])[data['valid'][index]]
        assert error.max() <= 1e-4, (shard, job, float(error.max()))
        current_errors.append(float(error.max()))
        targets = np.asarray([sequence[int(t)].get_bounding_box() for t in frames])
        gt_valid = np.isfinite(targets).all(-1) & (targets[:, 2:] > targets[:, :2]).all(-1)
        labels = data['history_iou'][index][past]
        assert (labels[~gt_valid] == -1).all()
        if gt_valid.any():
            measured_history = overlaps(data['history_boxes'][index][past][gt_valid].astype(np.float64), targets[gt_valid])
            history_error = np.abs(measured_history - labels[gt_valid])
            assert history_error.max() <= 1e-4, (shard, job, float(history_error.max()))
            history_errors.append(float(history_error.max()))
    if arrays:
        assert data.keys() == arrays[0].keys()
        assert all(data[key].dtype == arrays[0][key].dtype and data[key].shape[1:] == arrays[0][key].shape[1:] for key in data)
        assert all(cfg[key] == configs[0][key] for key in common)
    arrays.append(data)
    configs.append(cfg)
    receipts.append(done)
    jobs.extend(cfg['jobs'])
    evidence.append({'root': str(root), 'clips': count, 'GT_current_iou_max_error': max(current_errors),
                     'GT_history_iou_max_error': max(history_errors), 'all_valid_past_frames_exact': True,
                     'artifact_sha256': {name: sha(root / name) for name in ('config.json', 'completion.json', 'samples.npz', 'job_completed.txt')}})
    print('SHARD_PASS', json.dumps({'shard': shard, 'clips': count}), flush=True)
assert jobs == expected_jobs and len(jobs) == expected_count
merged = {key: np.concatenate([data[key] for data in arrays]) for key in arrays[0]}
out.mkdir(parents=True)
(out / 'jobs.json').write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
config = configs[0] | {'output': str(out), 'clips': expected_count, 'jobs': jobs,
                        'jobs_file': str(out / 'jobs.json'), 'source_configs': configs,
                        'merged_from': [item['root'] for item in evidence], 'merge_method': 'CPU exact ordered concatenation; no neural rerun'}
(out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
np.savez_compressed(out / 'samples.npz', **merged)
with np.load(out / 'samples.npz', allow_pickle=False) as archive:
    assert set(archive.files) == set(merged) and all(np.array_equal(archive[key], merged[key]) for key in merged)
done = {'completed': True, 'partition': args.partition, 'clips': expected_count,
        'valid_actions': int(merged['action_valid'].sum()), 'source_receipts': receipts,
        'decision_input_contains_future': False, 'official_tracking_accuracy': False,
        'cpu_merge_only': True, 'strict_npz_reload_equal': True}
(out / 'completion.json').write_text(json.dumps(done, indent=2) + '\n')
(out / 'job_completed.txt').write_text('Actual TRAIN-data CPU audit/merge complete; no tracking score claim.\n')
assert sha(parent) == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
receipt = {'status': 'PASS', 'partition': args.partition, 'clips': expected_count, 'source_GT_evidence': evidence,
           'all_ordered_jobs_exact': True, 'all_arrays_strict_reload_exact': True, 'parent_sha256': sha(parent),
           'decision_fields': decision_fields, 'causal_prefix_and_copy_before_future': True, 'prefix_epoch': 4, 'future_epoch': 4,
           'schema': {key: {'shape': list(value.shape), 'dtype': value.dtype.str} for key, value in merged.items()},
           'completed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
           'artifact_sha256': {name: sha(out / name) for name in ('config.json', 'completion.json', 'samples.npz', 'job_completed.txt')},
           'review_independence': 'same-family', 'acceptance_status': 'provisional',
           'execution': {'GPU_queries': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0, 'native_test_reads': False, 'runtime_seconds': time.perf_counter() - started},
           'limitation': 'Current and complete valid-past IoUs independently replayed against TRAIN GT; future IoUs are finite/range/provenance checked source-linked teacher outputs, not independently rerun.'}
(out / 'partition_cpu_acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'partition': args.partition, 'clips': expected_count, 'root': str(out)}), flush=True)
