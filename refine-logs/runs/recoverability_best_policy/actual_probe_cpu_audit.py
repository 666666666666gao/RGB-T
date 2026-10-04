"""Audit real old4 own-policy probes against TRAIN GT; never execute a neural model."""
import ast
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped

assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and Path.cwd() == Path('/data/gb/GOLA')
setup = Path('/data/gb/setup')
cache = Path('/data/gb/outputs/recoverability_current_policy_merged_20261004/own')
parent = Path('/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth')
started = time.perf_counter()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
review = read(setup / 'best_policy_collect_source_review_20261004.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
assert sha(parent) == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
tree = ast.parse(Path('research/recoverability_modules.py').read_text())
decision_fields = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                      and any(isinstance(target, ast.Name) and target.id == 'DECISION_FIELDS' for target in node.targets)))
assert not set(decision_fields) & {'current_iou', 'future_iou', 'history_iou', 'wrong_update_fraction', 'action_valid'}
reference = read(cache / 'train/config.json')
split = read(reference['split'])
assert len(split['train']) == 881 and len(split['validation']) == 98 and not set(split['train']) & set(split['validation'])
dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(reference['root'], reference['cache'])
sequences = {dataset[index].get_name(): dataset[index] for index in range(len(dataset))}
arms = {}
for part in ('train', 'validation'):
    root = Path('/data/gb/outputs') / ('recoverability_best_policy_collect_' + part + '_probe_20261004')
    cfg, done = read(root / 'config.json'), read(root / 'completion.json')
    jobs = read(setup / ('best_policy_jobs_' + part + '_probe_20261004.json'))['jobs']
    previous = read(cache / part / 'config.json')
    assert cfg['jobs'] == jobs == previous['jobs'][:16]
    assert (root / 'job_completed.txt').is_file() and done['completed']
    assert cfg['clips'] == done['clips'] == len(jobs) == 16 and cfg['partition'] == done['partition'] == part
    assert cfg['prefix_model'] == str(parent) and cfg['prefix_checkpoint_epoch'] == cfg['future_checkpoint_epoch'] == 4
    assert cfg['future_policy_mode'] == 'own' and cfg['prefix_write_verification'] == 'action'
    assert cfg['prefix_execution'] == 'per-sequence' and cfg['future_execution'] == 'per-action serial, identical deployed step and write verification'
    assert cfg['future_horizon'] == 3 and cfg['max_prefix'] == 1024 and cfg['batch_clips'] == 16 and cfg['forward_batch'] == 64
    assert cfg['decision_inputs_copied_before_future_decode'] and not done['decision_input_contains_future']
    assert not done['official_tracking_accuracy'] and cfg['seed'] == 42
    for key in ('root', 'cache', 'head', 'pretrained', 'motion_run', 'split', 'regions'):
        assert cfg[key] == previous[key]
    assert {job['sequence'] for job in jobs} <= set(split[part])
    with np.load(root / 'samples.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert set(decision_fields) <= set(arrays) and all(len(value) == 16 and np.isfinite(value).all() for value in arrays.values())
    assert arrays['instance_features'].shape == (16, 7, 5, 2, 768)
    assert arrays['image_boxes'].shape == (16, 7, 5, 4) and arrays['current_iou'].shape == arrays['valid'].shape == (16, 7, 5)
    assert arrays['action_valid'].shape == (16, 7, 5, 2) and arrays['future_iou'].shape == (16, 7, 5, 2, 3)
    assert np.array_equal(arrays['action_valid'][..., 0], arrays['valid'])
    assert np.array_equal(arrays['action_valid'][..., 1], arrays['valid'] & (arrays['raw_score'] > .84))
    errors = []
    for index, job in enumerate(jobs):
        frame = job['query_frame']
        valid_past = arrays['history_valid'][index].astype(bool)
        assert (arrays['history_frames'][index][valid_past] < frame).all()
        assert valid_past.any() and arrays['history_frames'][index][valid_past].max() == frame - 1
        gt = sequences[job['sequence']][frame].get_bounding_box()
        assert np.isfinite(gt).all() and (gt[2:] > gt[:2]).all()
        boxes = arrays['image_boxes'][index].astype(np.float64)
        intersection = np.maximum(np.minimum(boxes[..., 2:], gt[2:]) - np.maximum(boxes[..., :2], gt[:2]), 0).prod(-1)
        area = np.maximum(boxes[..., 2:] - boxes[..., :2], 0).prod(-1)
        measured = intersection / (area + (gt[2:] - gt[:2]).prod() - intersection)
        error = np.abs(measured - arrays['current_iou'][index])[arrays['valid'][index]]
        assert error.max() <= 1e-4, (part, job, float(error.max()))
        errors.append(float(error.max()))
    with np.load(cache / part / 'samples.npz', allow_pickle=False) as old:
        changed = {key: int(np.any(arrays[key] != old[key][:16], axis=tuple(range(1, arrays[key].ndim))).sum())
                   for key in ('history_boxes', 'image_boxes', 'current_iou', 'future_iou', 'prefix_counts')}
    arms[part] = {'status': 'PASS', 'root': str(root), 'clips': 16, 'past_causal': True,
                  'GT_current_iou_max_error': max(errors), 'changed_queries_against_same16_own25': changed,
                  'completion': done, 'artifact_sha256': {name: sha(root / name) for name in ('config.json', 'completion.json', 'samples.npz', 'job_completed.txt')}}
assert sha(parent) == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
record = {'status': 'PASS', 'both_partitions_passed': True, 'clips_per_partition': 16,
          'prefix_epoch': 4, 'future_teacher_is_global_best': True,
          'accepted_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'arms': arms,
          'review_independence': 'same-family', 'acceptance_status': 'provisional',
          'scope': 'Actual32 TRAIN-partition collection probe and currentGT replay; no method improvement or native result claim.',
          'execution': {'neural_forward_calls': 0, 'optimizer_steps': 0, 'GPU_queries': 0, 'native_test_reads': False,
                        'runtime_seconds': time.perf_counter() - started}}
out = setup / 'best_policy_probe_actual_cpu_acceptance_20261004.json'
out.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': 'PASS', 'receipt': str(out)}))
