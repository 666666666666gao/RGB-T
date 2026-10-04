"""Read audited cached outputs and TRAIN-heldout labels; no neural execution."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
SETUP = Path('/data/gb/setup')
CACHE = Path('/data/gb/outputs/recoverability_current_policy_merged_20261004')
OUTPUTS = Path('/data/gb/outputs')
rows, inputs, vectors = {}, {}, {}
job_reference = None
for policy in ('c1', 'own'):
    cache = CACHE / policy / 'validation'
    jobs = json.loads((cache / 'jobs.json').read_text())['jobs']
    assert len(jobs) == 128
    if job_reference is None:
        job_reference = jobs
    assert jobs == job_reference
    with np.load(cache / 'samples.npz', allow_pickle=False) as data:
        values = {key: data[key] for key in (
            'current_iou', 'valid', 'original_choice', 'action_valid',
            'future_iou', 'wrong_update_fraction')}
    assert values['current_iou'].shape == (128, 7, 5)
    valid = values['valid']
    quality = values['current_iou']
    keep = values['original_choice']
    index = np.arange(128)
    baseline = quality[index, 0, keep]
    local_best = np.where(valid[:, 0], quality[:, 0], -1).max(1)
    all_best = np.where(valid, quality, -1).max((1, 2))
    local_recall, any_recall = local_best >= .5, all_best >= .5
    utility = (.7 * quality[..., None] + .3 * values['future_iou'].mean(-1)
               - .1 * values['wrong_update_fraction'])
    inputs[policy] = {'queries': 128, 'baseline_failed_queries': int((baseline < .2).sum()),
                     'local_correct_candidate_queries': int(local_recall.sum()),
                     'any_region_correct_candidate_queries': int(any_recall.sum()),
                     'local_missing_but_some_extra_region_has_correct_candidate': int((~local_recall & any_recall).sum()),
                     'baseline_failed_with_local_correct_candidate': int(((baseline < .2) & local_recall).sum()),
                     'baseline_failed_local_missing_extra_can_supply_correct': int(((baseline < .2) & ~local_recall & any_recall).sum())}
    for mode, stem in [('oracle', 'current_policy_fit'), ('selector', 'current_policy_selector_fit')]:
        label = mode + '_' + policy
        root = OUTPUTS / ('recoverability_' + stem + '_' + policy + '_b384_full_20261004')
        fit = json.loads((SETUP / (stem + '_actual_full_fit_cpu_audit_20261004.json')).read_text())
        assert fit['status'] == 'PASS'
        arm = fit['arms'][policy]
        assert arm['all21_cached_fields_CPU_reproduced']
        path = root / 'best_validation.npz'
        with np.load(path, allow_pickle=False) as saved:
            result = {key: saved[key] for key in saved.files}
        selected = quality[index, result['region'], result['candidate']]
        assert np.array_equal(selected, result['current'])
        assert np.array_equal(baseline, result['c1_current'])
        allowed = np.zeros((128, 7), dtype=bool)
        allowed[:, 0] = True
        allowed[index, result['searched_region']] = result['search_triggered']
        budget_best = np.where(valid & allowed[..., None], quality, -1).max((1, 2))
        budget_recall = budget_best >= .5
        assert np.array_equal(local_recall, result['original_recall'])
        assert np.array_equal(budget_recall, result['budget_recall'])
        selected_utility = utility.reshape(128, -1)[index, result['flat_action']]
        assert np.allclose(selected_utility - .01 * result['search_triggered'], result['utility'], atol=2e-7, rtol=0)
        budget_action_utility = np.where(values['action_valid'] & allowed[..., None, None], utility, -np.inf).reshape(128, -1)
        best_utility_action = budget_action_utility.argmax(1)
        best_utility = budget_action_utility[index, best_utility_action]
        best_utility_region = best_utility_action // 10
        best_utility_candidate = best_utility_action % 10 // 2
        best_utility_pause = best_utility_action % 2
        utility_regret = best_utility - selected_utility
        assert (utility_regret >= -2e-7).all()
        delta = selected - baseline
        found = ~local_recall & budget_recall
        correct = selected >= .5
        records = []
        for i, job in enumerate(jobs):
            if delta[i] != 0 or baseline[i] < .2 or found[i]:
                records.append(dict(job, cached_query_index=i, baseline_iou=float(baseline[i]),
                                    selected_iou=float(selected[i]), local_best_iou=float(local_best[i]),
                                    actual_budget_best_iou=float(budget_best[i]), any_region_best_iou=float(all_best[i]),
                                    selected_gt_utility_before_search_cost=float(selected_utility[i]),
                                    actual_budget_oracle_gt_utility_before_search_cost=float(best_utility[i]),
                                    actual_budget_gt_utility_regret=float(utility_regret[i]),
                                    actual_budget_best_utility_region=int(best_utility_region[i]),
                                    actual_budget_best_utility_candidate=int(best_utility_candidate[i]),
                                    actual_budget_best_utility_pause=bool(best_utility_pause[i]),
                                    actual_budget_best_utility_current_iou=float(quality[i, best_utility_region[i], best_utility_candidate[i]]),
                                    search_triggered=bool(result['search_triggered'][i]),
                                    selected_region=int(result['region'][i]), selected_candidate=int(result['candidate'][i]),
                                    pause=bool(result['pause'][i])))
        rows[label] = {'best_epoch': arm['best_epoch'], 'source_saved_output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'current_iou_mean': float(selected.astype(np.float64).mean()),
                       'baseline_current_iou_mean': float(baseline.astype(np.float64).mean()),
                       'current_gain_percentage_points': float(delta.astype(np.float64).mean() * 100),
                       'improved_queries': int((delta > 0).sum()), 'worsened_queries': int((delta < 0).sum()),
                       'unchanged_queries': int((delta == 0).sum()),
                       'actual_extra_searches': int(result['search_triggered'].sum()),
                       'added_correct_candidate_queries': int(found.sum()),
                       'added_correct_candidate_and_selected_correct': int((found & correct).sum()),
                       'budget_has_correct_candidate_but_selected_incorrect': int((budget_recall & ~correct).sum()),
                       'selected_failed_queries': int((selected < .2).sum()),
                       'actual_budget_gt_utility_regret_above_point03_queries': int((utility_regret > .03).sum()),
                       'actual_budget_gt_utility_regret_mean': float(utility_regret.astype(np.float64).mean()),
                       'positive_delta_sum': float(delta[delta > 0].astype(np.float64).sum()),
                       'negative_delta_sum': float(delta[delta < 0].astype(np.float64).sum()),
                       'diagnostic_query_records': records}
        vectors[label] = result
pair_fields = ['current', 'flat_action', 'search_triggered', 'searched_region', 'pause']
pairs = {}
for left, right in [('oracle_c1', 'selector_c1'), ('oracle_own', 'selector_own'), ('selector_c1', 'selector_own')]:
    pairs[left + '_vs_' + right] = {key: int((vectors[left][key] != vectors[right][key]).sum()) for key in pair_fields}
record = {'status': 'ACTUAL_AUDITED_CACHE_OPPORTUNITY_DIAGNOSIS', 'observed_at': datetime.now().astimezone().isoformat(),
          'scope': '128 cached TRAIN-heldout queries, no continuous-video or native-test result',
          'thresholds': {'failed_iou_less_than': .2, 'correct_candidate_iou_at_least': .5},
          'inputs': inputs, 'cells': rows, 'pair_output_difference_query_counts': pairs,
          'limitations': ['Ground truth candidate maxima are diagnostic upper bounds, not deployable search decisions.',
                         'Different future teachers have different utility labels; current-IoU comparisons use matched inputs.',
                         'These queries selected checkpoints and are not independent confirmation.',
                         'No architecture change or performance attribution before actual full98 closure.'],
          'execution': {'neural_forwards': 0, 'GPU_queries': 0, 'weights_modified': False}}
path = SETUP / 'current_policy_actual_cached_opportunity_diagnosis_20261004.json'
path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': record['status'], 'inputs': inputs,
                  'cells': {k: {f: v[f] for f in ['improved_queries', 'worsened_queries', 'unchanged_queries', 'actual_extra_searches',
                                                    'added_correct_candidate_queries', 'added_correct_candidate_and_selected_correct',
                                                    'budget_has_correct_candidate_but_selected_incorrect', 'current_gain_percentage_points']}
                            for k, v in rows.items()}, 'pairs': pairs}, indent=2))
