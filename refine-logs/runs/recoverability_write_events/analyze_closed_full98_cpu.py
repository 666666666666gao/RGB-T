"""Analyze already completed internal scores; no inference or native-test selection."""
import csv
import json
import random
import statistics
from pathlib import Path

folder = Path(__file__).resolve().parent
intake = json.loads((folder / 'actual_all16_closed_full98_intake.json').read_text())
reference_path = folder.parent / 'recoverability_write_pair/joint_report/per_sequence.csv'
reference = {}
with reference_path.open(newline='', encoding='utf-8') as stream:
    for row in csv.DictReader(stream):
        if row['variant'] in ('baseline', 'c1', 'write_pair_reference_own_b384'):
            reference.setdefault(row['variant'], {})[row['sequence']] = float(row['mean_valid_iou'])
names = sorted(reference['baseline'])
assert len(names) == 98 and all(sorted(values) == names for values in reference.values())
reference_report = json.loads(reference_path.with_name('full_recoverability_report.json').read_text())
assert all(abs(statistics.fmean(values.values()) - reference_report['variants'][label]['sequence_mean_iou']) < 1e-12
           for label, values in reference.items())
rng = random.Random(42)
draws = [[rng.randrange(98) for _ in names] for _ in range(5000)]


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = int(position)
    weight = position - lower
    return values[lower] * (1 - weight) + values[lower + 1] * weight


summary, per_sequence = [], []
for candidate in intake['all16']:
    score = candidate['score']
    current = {row['sequence']: row['mean_iou'] for row in score['per_sequence']}
    assert sorted(current) == names
    assert abs(statistics.fmean(current.values()) - candidate['sequence_mean_iou']) < 1e-12
    row = {'arm': candidate['arm'], 'epoch': candidate['epoch'], 'model': candidate['model'],
           'selected_by_original_internal_queue': candidate['model'] == intake['selection']['checkpoint'],
           'sequence_mean_iou': candidate['sequence_mean_iou'], 'paired': {}}
    for label, values in reference.items():
        delta = [100 * (current[name] - values[name]) for name in names]
        resamples = sorted(statistics.fmean(delta[index] for index in draw) for draw in draws)
        row['paired'][label] = {'delta_percentage_points': statistics.fmean(delta),
                                'paired_sequence_95_interval_percentage_points':
                                [percentile(resamples, .025), percentile(resamples, .975)],
                                'improved_sequences': sum(value > 1e-12 for value in delta),
                                'worsened_sequences': sum(value < -1e-12 for value in delta),
                                'tied_sequences': sum(abs(value) <= 1e-12 for value in delta)}
    summary.append(row)
    if row['selected_by_original_internal_queue']:
        for name in names:
            per_sequence.append({'sequence': name, 'new_mean_iou': current[name],
                                 'baseline_mean_iou': reference['baseline'][name],
                                 'old4_mean_iou': reference['write_pair_reference_own_b384'][name],
                                 'delta_vs_baseline_percentage_points': 100 * (current[name] - reference['baseline'][name]),
                                 'delta_vs_old4_percentage_points': 100 * (current[name] - reference['write_pair_reference_own_b384'][name])})
assert len(summary) == 16 and len(per_sequence) == 98
selected = next(row for row in summary if row['selected_by_original_internal_queue'])
ordered = sorted(per_sequence, key=lambda row: row['delta_vs_old4_percentage_points'])
result = {'completed': True, 'source': 'Already completed full98 TRAIN-GT scores and existing actual-GT reference per_sequence.csv.',
          'neural_forward_calls': 0, 'optimizer_steps': 0, 'native_test_data_read': False,
          'formal_native_metrics_completed': False, 'models': summary, 'selected': selected,
          'selected_worst10_vs_old4': ordered[:10], 'selected_best10_vs_old4': ordered[-10:][::-1],
          'all16_below_protected_old4': all(row['paired']['write_pair_reference_own_b384']['delta_percentage_points'] < 0 for row in summary),
          'bootstrap': {'resamples': 5000, 'seed': 42, 'rng': 'Python stdlib MT19937; paired identical sequence draws across candidates/references.',
                        'scope': 'Descriptive fixed-checkpoint developer-sequence uncertainty; reused selection data, no untouched confirmation, no seed-stability evidence.'}}
(folder / 'actual_all16_full98_paired_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
for name, rows in (('actual_selected_full98_per_sequence.csv', per_sequence),
                   ('actual_all16_full98_table.csv', [{key: row[key] for key in ('arm', 'epoch', 'model', 'selected_by_original_internal_queue', 'sequence_mean_iou')} |
                                                    {f'delta_vs_{label}_percentage_points': row['paired'][label]['delta_percentage_points'] for label in reference}
                                                    for row in summary])):
    with (folder / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({'selected': selected, 'worst3': ordered[:3], 'best3': ordered[-3:][::-1],
                  'all16_below_old4': result['all16_below_protected_old4']}))
