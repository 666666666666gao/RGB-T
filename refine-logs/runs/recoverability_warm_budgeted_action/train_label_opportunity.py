"""Count training-label opportunities under the existing seven search budgets; no NN."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
sys.path[:0] = ['/data/gb/experiments/recoverability_warm_budgeted_action_20261004', '/data/gb/GOLA']
import numpy as np
import torch
from research.recoverability_modules import action_utility

assert not torch.cuda.is_initialized()
root = Path('/data/gb/outputs/recoverability_future_policy_merged_20261004')
result = {'status': 'TRAIN_LABEL_DIAGNOSTIC_ONLY',
          'observed_cst': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
          'scope': '902 matched TRAIN queries per future teacher; oracle labels only, not predicted accuracy or causal online benefit',
          'decision_threshold': .03, 'one_extra_search_cost': .01, 'NN_forwards': 0,
          'GPU_queries': 0, 'official_test_read': False, 'validation_read': False,
          'training_or_model_modified': False, 'arms': {}}
for teacher in ['c1', 'own']:
    path = root / teacher / 'train'
    with np.load(path / 'samples.npz', allow_pickle=False) as z:
        data = {key: torch.from_numpy(z[key].copy()) for key in
                ['current_iou', 'future_iou', 'wrong_update_fraction', 'action_valid', 'original_choice', 'valid']}
    receipt = json.loads((path / 'completion.json').read_text())
    assert receipt['completed'] and receipt['clips'] == 902
    utility, legal = action_utility(data).flatten(1), data['action_valid'].flatten(1)
    current = data['current_iou'].float().flatten(1)
    keep = data['original_choice'].long()
    rows, indices = torch.arange(len(keep)), torch.arange(70)
    reference, regions = utility[rows, keep * 2], indices // 10
    adjusted = utility - reference[:, None] - .01 * (regions > 0)[None] - .03 * (indices[None] != (keep * 2)[:, None])
    quality = data['current_iou'].float()
    local_recall = quality[:, 0].masked_fill(~data['valid'][:, 0], -1).max(-1).values >= .5
    keep_quality = quality[rows, 0, keep]
    sets, any_gain = [], torch.zeros(len(keep), dtype=torch.bool)
    for region in range(7):
        available = legal & ((regions == 0) | (regions == region))[None]
        gain, winner = adjusted.masked_fill(~available, -torch.inf).max(1)
        winner = torch.where(gain > 0, winner, keep * 2)
        positive = winner != keep * 2
        any_gain |= positive
        chosen_quality = current[rows, winner // 2]
        sets.append({'budget_region': region, 'queries': len(keep),
                     'oracle_changes': int(positive.sum()), 'oracle_location_changes': int((winner // 2 != keep).sum()),
                     'oracle_pauses': int((winner % 2 == 1).sum()), 'oracle_extra_selected': int((winner >= 10).sum()),
                     'same_state_rescue': int(((keep_quality < .2) & (chosen_quality >= .5)).sum()),
                     'same_state_harm': int(((keep_quality >= .5) & (chosen_quality < .2)).sum()),
                     'mean_net_label_gain': float(gain.clamp(min=0).mean()),
                     'positive_mean_net_label_gain': float(gain[positive].mean()) if positive.any() else None})
    result['arms'][teacher] = {'queries': len(keep), 'baseline_correct': int((keep_quality >= .5).sum()),
                              'baseline_failure': int((keep_quality < .2).sum()), 'local_recall': int(local_recall.sum()),
                              'failed_but_local_missing': int(((keep_quality < .2) & ~local_recall).sum()),
                              'any_budget_label_advantage_queries': int(any_gain.sum()),
                              'oracle_positive_sets': sum(row['oracle_changes'] for row in sets),
                              'all_legal_budget_sets': 902 * 7, 'per_budget': sets}
assert not torch.cuda.is_initialized()
out = Path('/data/gb/setup/warm_budgeted_action_train_label_opportunity_20261004.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'path': str(out), 'arms':
                  {key: {k: v for k, v in arm.items() if k != 'per_budget'} for key, arm in result['arms'].items()}}))
