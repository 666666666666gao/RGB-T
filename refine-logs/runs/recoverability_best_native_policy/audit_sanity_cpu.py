"""Accept four actual completed old4 collections against TRAIN GT before full collection."""
import json
import os
from pathlib import Path

import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
setup = Path('/data/gb/setup')
read = lambda path: json.loads(Path(path).read_text())
prepared = read(setup / 'best_native_policy_jobs_cpu_acceptance_20261005.json')
assert prepared['status'] == 'PASS'
reference = prepared['reference']
dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(reference['root'], reference['cache'])
sequences = {dataset[i].get_name(): dataset[i] for i in range(len(dataset))}
evidence = []
for gpu in range(4):
    root = Path(f'/data/gb/outputs/recoverability_best_native_policy_collect_sanity_gpu{gpu}_20261005')
    assert (root / 'job_completed.txt').is_file()
    config, receipt = read(root / 'config.json'), read(root / 'completion.json')
    jobs = read(prepared['shards'][f'sanity_gpu{gpu}']['jobs_file'])['jobs']
    assert config['jobs'] == jobs and config['partition'] == receipt['partition'] == 'train'
    assert config['prefix_model'] == prepared['teacher'] and config['prefix_checkpoint_epoch'] == config['future_checkpoint_epoch'] == 4
    assert config['future_policy_mode'] == 'own' and config['prefix_write_verification'] == 'action'
    assert receipt['completed'] and receipt['clips'] == 2 and not receipt['decision_input_contains_future']
    with np.load(root / 'samples.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert all(len(value) == 2 and np.isfinite(value).all() for value in data.values())
    assert np.array_equal(data['action_valid'][..., 0], data['valid'])
    assert np.array_equal(data['action_valid'][..., 1], data['valid'] & (data['raw_score'] > .84))
    errors = []
    for row, job in enumerate(jobs):
        target = sequences[job['sequence']][job['query_frame']].get_bounding_box()
        boxes = data['image_boxes'][row].astype(np.float64)
        intersection = np.maximum(np.minimum(boxes[..., 2:], target[2:]) - np.maximum(boxes[..., :2], target[:2]), 0).prod(-1)
        overlap = intersection / (np.maximum(boxes[..., 2:] - boxes[..., :2], 0).prod(-1) + np.prod(target[2:] - target[:2]) - intersection)
        error = np.abs(overlap - data['current_iou'][row])[data['valid'][row]]
        assert error.max() <= 1e-4
        past = data['history_frames'][row][data['history_valid'][row]]
        assert np.array_equal(past, np.arange(job['query_frame']))
        errors.append(float(error.max()))
    evidence.append({'gpu': gpu, 'root': str(root), 'clips': 2, 'GT_current_iou_max_error': max(errors),
                     'elapsed_seconds': receipt['elapsed_seconds'], 'peak_cuda_mib': receipt['peak_cuda_mib']})
out = setup / 'best_native_policy_sanity_cpu_acceptance_20261005.json'
assert not out.exists()
record = {'status': 'PASS', 'all_four_collectors_completed': True, 'actual_queries': 8,
          'evidence': evidence, 'native_accuracy_completed': False, 'training_completed': False,
          'scope': 'Actual finite collection arrays, causal prefix indices and current TRAIN GT; future labels source-linked, not independently rerun.'}
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
