"""CPU-only TRAIN shard0 comparison; no model, validation, or native test reads."""
import ast
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

START = time.perf_counter()
BASE = Path('/data/gb/outputs')
OLD = BASE / 'recoverability_future_c1_train_shard0_20261004'
NEW = BASE / 'recoverability_current_policy_c1_train_shard0_20261004'
OWN = BASE / 'recoverability_current_policy_own_train_shard0_20261004'
OUT = Path('/data/gb/setup/current_policy_recollection_actual_shard0_state_shift_20261004.json')


def load(root):
    assert (root / 'job_completed.txt').is_file()
    config = json.loads((root / 'config.json').read_text())
    receipt = json.loads((root / 'completion.json').read_text())
    assert config['partition'] == receipt['partition'] == 'train'
    assert receipt['completed'] and not receipt['decision_input_contains_future']
    assert not receipt['official_tracking_accuracy']
    assert config['clips'] == receipt['clips'] == len(config['jobs']) == 225
    with np.load(root / 'samples.npz') as archive:
        data = {key: archive[key].copy() for key in archive.files}
    assert all(len(value) == 225 and np.isfinite(value).all() for value in data.values())
    query = np.asarray([job['query_frame'] for job in config['jobs']])[:, None]
    assert ((data['history_frames'] < query) | ~data['history_valid']).all()
    return config, data


def different_rows(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    return (left != right).reshape(225, -1).any(1)


old_config, old = load(OLD)
new_config, new = load(NEW)
own_config, own = load(OWN)
assert old_config['prefix_checkpoint_epoch'] == 4
assert new_config['prefix_checkpoint_epoch'] == own_config['prefix_checkpoint_epoch'] == 25
assert old_config['jobs'] == new_config['jobs'] == own_config['jobs']
assert old.keys() == new.keys() == own.keys()
future_fields = {'future_iou', 'wrong_update_fraction'}
assert all(np.array_equal(new[key], own[key]) for key in new if key not in future_fields)
tree = ast.parse(Path('/data/gb/experiments/recoverability_current_policy_fit_20261004/research/recoverability_modules.py').read_text())
fields = next(ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DECISION_FIELDS' for t in node.targets))
decision_changes = {key: int(different_rows(old[key], new[key]).sum()) for key in fields}
history_fields = ('history_boxes', 'history_frames', 'history_write', 'history_valid', 'history_instance_descriptors')
history_changed = np.logical_or.reduce([different_rows(old[key], new[key]) for key in history_fields])
local_fields = ('features', 'evidence', 'raw_score', 'boxes', 'instance_features', 'valid', 'c1_quality')
local_changed = np.logical_or.reduce([different_rows(old[key][:, 0], new[key][:, 0]) for key in local_fields])
local_changed |= different_rows(old['original_choice'], new['original_choice'])
rows = np.arange(225)
keep_iou = {name: data['current_iou'][rows, 0, data['original_choice'].astype(np.int64)]
            for name, data in [('old4_prefix', old), ('own25_prefix', new)]}
result = {
    'status': 'ACTUAL_MATCHED_TRAIN_SHARD0_DIAGNOSTIC_COMPLETE',
    'observed_at': datetime.now().astimezone().isoformat(),
    'scope': '225 matched TRAIN queries only; incomplete 902-query collection, no validation or native test reads',
    'input_roots': [str(OLD), str(NEW), str(OWN)],
    'matched_nonfuture_arrays_exact': True,
    'nonfuture_array_count': len(set(new) - future_fields),
    'decision_field_changed_query_counts': decision_changes,
    'history_changed_queries': int(history_changed.sum()),
    'local_visual_candidates_changed_queries': int(local_changed.sum()),
    'prefix_counter_columns': new_config['prefix_counts_columns'],
    'prefix_counter_sums': {'old4': old['prefix_counts'].sum(0).tolist(), 'own25': new['prefix_counts'].sum(0).tolist()},
    'local_c1_quality_under_each_prefix': {
        name: {'known_queries': int((value >= 0).sum()), 'mean_iou': float(value[value >= 0].mean()),
               'failed_queries': int(((value >= 0) & (value < .2)).sum()), 'correct_queries': int((value >= .5).sum())}
        for name, value in keep_iou.items()},
    'future_label_differences_between_current_teachers': {
        key: {'different_elements': int(np.count_nonzero(new[key] != own[key])),
              'max_absolute_difference': float(np.abs(new[key] - own[key]).max())}
        for key in sorted(future_fields)},
    'limits': 'Input differences include changed motion proposals as well as history. History and local-candidate counts are reported separately. Query IoU is a prefix-state diagnostic, not independent model tracking accuracy or causal intervention benefit.',
    'neural_inference_runs': 0,
    'gpu_queries': 0,
    'runtime_seconds': time.perf_counter() - START,
}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2, allow_nan=False))
