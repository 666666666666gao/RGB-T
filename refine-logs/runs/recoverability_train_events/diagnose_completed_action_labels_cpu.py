"""Describe already accepted TRAIN labels; oracle label availability is not online accuracy."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


root = Path('/data/gb/GOLA')
folder = root / 'refine-logs/runs/recoverability_train_events'
plan = json.loads((folder / 'plan.json').read_text())
gate = json.loads(Path('/data/gb/setup/train_events_full_cpu_acceptance_20261005.json').read_text())
assert gate['status'] == 'PASS' and gate['new_clips'] == 716
assert gate['GT_current_and_all_valid_history_verified']
inputs = [('base', Path(plan['base_train']), 1762)] + [
    ('new_events', Path(plan['full'][str(gpu)]['output']), 179) for gpu in range(4)]
blocks = {'base': [], 'new_events': []}
keys = ['current_iou', 'future_iou', 'wrong_update_fraction', 'action_valid', 'valid', 'original_choice']
for group, path, count in inputs:
    assert (path / 'job_completed.txt').is_file()
    with np.load(path / 'samples.npz', allow_pickle=False) as archive:
        block = {key: archive[key] for key in keys}
    assert block['current_iou'].shape == (count, 7, 5)
    assert block['future_iou'].shape == (count, 7, 5, 2, 3)
    assert block['action_valid'].shape == (count, 7, 5, 2)
    blocks[group].append(block)

results = {}
for group, parts in blocks.items():
    data = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    current = data['current_iou'].astype(np.float32)
    # Exactly the scalar utility coefficients in research/recoverability_modules.py::action_utility.
    utility = (.7 * current[..., None] + .3 * data['future_iou'].astype(np.float32).mean(-1)
               - .1 * data['wrong_update_fraction'].astype(np.float32))
    legal = data['action_valid']
    rows = np.arange(len(current))
    keep = data['original_choice']
    assert legal[rows, 0, keep, 0].all()
    reference = utility[rows, 0, keep, 0]
    delta = utility - reference[:, None, None, None]
    best_local = np.where(legal[:, 0], utility[:, 0], -np.inf).max(axis=(1, 2))
    best_extra = np.where(legal[:, 1:], utility[:, 1:] - .01, -np.inf).max(axis=(1, 2, 3))
    local_quality = np.where(data['valid'][:, 0], current[:, 0], -1).max(-1)
    extra_quality = np.where(data['valid'][:, 1:], current[:, 1:], -1).max(axis=(1, 2))
    selected_quality = current[rows, 0, keep]
    paired = legal[..., 1]
    pause_gain = utility[..., 1] - utility[..., 0]
    results[group] = {
        'queries': len(current),
        'original_failed_queries_iou_lt_02': int((selected_quality < .2).sum()),
        'original_correct_queries_iou_ge_05': int((selected_quality >= .5).sum()),
        'original_candidate_present_iou_ge_05': int((local_quality >= .5).sum()),
        'original_candidate_absent_iou_lt_05': int((local_quality < .5).sum()),
        'absent_correct_candidate_reintroduced_by_any_cached_extra_region': int(((local_quality < .5) & (extra_quality >= .5)).sum()),
        'queries_with_local_oracle_utility_gain_gt_005': int((best_local - reference > .05).sum()),
        'queries_with_extra_oracle_gain_gt_005_over_best_local_after_001_cost': int((best_extra - best_local > .05).sum()),
        'queries_with_any_legal_action_utility_gain_gt_005': int((legal & (delta > .05)).any(axis=(1, 2, 3)).sum()),
        'queries_with_any_harmful_legal_action_utility_loss_gt_005': int((legal & (delta < -.05)).any(axis=(1, 2, 3)).sum()),
        'queries_with_valid_pause_write_pair': int(paired.any(axis=(1, 2)).sum()),
        'queries_with_pause_utility_gain_gt_001': int((paired & (pause_gain > .01)).any(axis=(1, 2)).sum()),
        'queries_with_pause_utility_loss_gt_001': int((paired & (pause_gain < -.01)).any(axis=(1, 2)).sum()),
        'mean_local_oracle_utility_gain': float((best_local - reference).mean()),
        'utility_definition': '.7*current_iou + .3*mean(future_iou over3) - .1*wrong_update_fraction',
        'new_online_accuracy_computed': False
    }
assert results['base']['queries'] == 1762 and results['new_events']['queries'] == 716
result = {
    'status': 'PASS', 'created_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'datasets_read': 'Previously completed TRAIN caches only; no VAL or native TEST reads.',
    'results': results,
    'interpretation_limit': 'GT-derived label inventory and oracle diagnostic. Regions were already collected; maxima are not deployed selector choices, realized online gains, or formal benchmark metrics.',
    'supervision_limit': gate['future_label_scope'],
    'execution': {'live_training_progress_queries': 0, 'model_imports': 0, 'neural_forward_calls': 0,
                  'optimizer_steps': 0, 'GPU_queries': 0, 'live_training_source_changed': False,
                  'npz_fields_loaded': keys}
}
output = folder / 'actual_completed_action_label_inventory.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, allow_nan=False), flush=True)
