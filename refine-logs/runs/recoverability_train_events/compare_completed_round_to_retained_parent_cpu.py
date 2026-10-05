"""Describe actual completed native regressions versus the retained parent; no model execution."""
import csv
import hashlib
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CURRENT = ROOT / 'refine-logs/runs/recoverability_best_native_policy/native_complete'
PARENT = ROOT / 'refine-logs/runs/recoverability_write_pair'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path, label):
    with path.open(encoding='utf-8', newline='') as stream:
        values = [row for row in csv.DictReader(stream) if row['variant'] == label]
    assert len({row['sequence'] for row in values}) == len(values)
    return {row['sequence']: row for row in values}


def main():
    complete_path = CURRENT / 'complete_report/complete_core_report.json'
    assert hashlib.sha256(complete_path.read_bytes()).hexdigest() == '860d21179810173d7d990f2a522c4217ef011e467e1dabe7b8d17cd8b024cd33'
    complete = read(complete_path)
    datasets = {}
    sources = {}
    for dataset, count, metrics in [('lasher', 245, ('PR', 'NPR', 'SR')), ('rgbt234', 234, ('MPR', 'MSR'))]:
        old = PARENT / ('own4_native_' + dataset)
        new = CURRENT / dataset
        old_report = read(old / 'core_report/full_report.json')
        new_report = complete['datasets'][dataset]['native']
        for key in ('sequences', 'frames', 'actual_data_root', 'official_gt_source', 'official_protocol'):
            assert old_report[key] == new_report[key], key
        old_csv, new_csv = old / 'core_report/per_sequence.csv', new / 'core_report/per_sequence.csv'
        parents = rows(old_csv, 'write_pair_reference_own_b384')
        children = rows(new_csv, 'best_native_policy_complete')
        assert set(parents) == set(children) and len(parents) == count
        for reference in ('baseline', 'c1'):
            left, right = rows(old_csv, reference), rows(new_csv, reference)
            assert set(left) == set(right) == set(parents)
            assert all(float(left[name][metric]) == float(right[name][metric]) for name in parents for metric in metrics)
        measured = {}
        for metric in metrics:
            deltas = [{'sequence': name, 'parent_percent': float(parents[name][metric]),
                       'completed_round_percent': float(children[name][metric]),
                       'delta_percentage_points': float(children[name][metric]) - float(parents[name][metric])}
                      for name in sorted(parents)]
            delta = new_report['variants']['best_native_policy_complete']['overall_metrics_percent'][metric] - old_report['variants']['write_pair_reference_own_b384']['overall_metrics_percent'][metric]
            assert abs(statistics.mean(row['delta_percentage_points'] for row in deltas) - delta) < 1e-10
            measured[metric] = {'overall_delta_percentage_points': delta,
                                'improved_sequences': sum(row['delta_percentage_points'] > 1e-12 for row in deltas),
                                'degraded_sequences': sum(row['delta_percentage_points'] < -1e-12 for row in deltas),
                                'unchanged_sequences': sum(abs(row['delta_percentage_points']) <= 1e-12 for row in deltas),
                                'largest_ten_degradations': sorted(deltas, key=lambda row: row['delta_percentage_points'])[:10],
                                'largest_ten_improvements': sorted(deltas, key=lambda row: -row['delta_percentage_points'])[:10],
                                'all_sequences': deltas}
        mechanism_path = old / 'mechanism_report/full_recoverability_report.json'
        old_mechanism = read(mechanism_path)['variants']['write_pair_reference_own_b384']
        new_mechanism = complete['datasets'][dataset]['mechanism']['variants']['best_native_policy_complete']
        mechanism = {}
        for label, value in [('retained_parent', old_mechanism), ('completed_round', new_mechanism)]:
            counter = value['counters']
            selected = {key: counter[key] for key in ('extra_visual_forwards', 'missing_correct_candidates_reintroduced',
                        'reintroduced_candidates_selected_correctly', 'same_state_rescues', 'same_state_harms',
                        'template_updates', 'wrong_template_updates_localization_proxy')}
            selected['failure_events'] = len(value['failure_events'])
            selected['correct_selection_given_reintroduction'] = counter['reintroduced_candidates_selected_correctly'] / counter['missing_correct_candidates_reintroduced']
            selected['wrong_template_write_rate_localization_proxy'] = value['rates']['wrong_template_write_rate']
            mechanism[label] = selected
        datasets[dataset] = {'sequences': count, 'paired_native_metrics': measured, 'descriptive_mechanisms': mechanism}
        for path in (old_csv, new_csv, old / 'core_report/full_report.json', mechanism_path):
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources[str(complete_path.relative_to(ROOT))] = hashlib.sha256(complete_path.read_bytes()).hexdigest()
    result = {'completed': True, 'created_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'datasets': datasets, 'source_sha256': sources,
              'scope': 'Compare already completed epoch40 native predictions to retained old4 parent on identical sequence sets, annotations, baseline and C1 metrics; no checkpoint selection or training changes.',
              'limitations': ['Mechanism trajectories differ between models; aggregated counters cannot establish same-state causal effects.',
                              'Formal benchmark results have informed development; this is descriptive diagnosis, not an untouched validation test.'],
              'execution': {'GPU_queries': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0, 'new_checkpoints': 0}}
    destination = HERE / 'actual_completed_native_parent_regression_diagnosis.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({dataset: {'deltas': {metric: row['overall_delta_percentage_points'] for metric, row in value['paired_native_metrics'].items()},
                               'mechanisms': value['descriptive_mechanisms']} for dataset, value in datasets.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
