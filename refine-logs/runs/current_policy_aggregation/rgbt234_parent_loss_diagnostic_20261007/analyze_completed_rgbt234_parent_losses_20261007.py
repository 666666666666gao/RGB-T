"""Describe already-completed RGBT234 sequence losses; no NN or re-scoring."""
import csv
import json
import math
from pathlib import Path

repo = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
root = repo / 'refine-logs/runs/current_policy_aggregation/complete/training'
dataset = root / 'native/rgbt234'
out = repo / 'refine-logs/runs/current_policy_aggregation/rgbt234_parent_loss_diagnostic_20261007'
assert not out.exists()
core = json.loads((dataset / 'core_report/full_report.json').read_text())
assert core['all_actual_ground_truth_verified'] and core['sequences'] == 234 and core['frames'] == 116649
plan = json.loads((root / 'native/plan.json').read_text())
lengths = plan['datasets']['rgbt234']['lengths']
with (dataset / 'gross_parent_paired_report/per_sequence.csv').open(encoding='utf-8-sig') as stream:
    source = list(csv.DictReader(stream))
labels = ('gross_parent', 'selected_ABC')
by_label = {label: {r['sequence']: r for r in source if r['variant'] == label} for label in labels}
assert len(source) == 468 and all(set(by_label[label]) == set(lengths) for label in labels)
assert len(lengths) == 234 and sum(lengths.values()) == 116649
rows = []
for name in sorted(lengths):
    old, current = (by_label[label][name] for label in labels)
    row = dict(sequence=name, frames=lengths[name])
    for metric in ('MPR', 'MSR'):
        row[metric + '_gross_parent'] = float(old[metric])
        row[metric + '_selected'] = float(current[metric])
        row[metric + '_delta_pp'] = float(current[metric]) - float(old[metric])
    for metric in ('valid_tracking_frames', 'failed_tracking_frames', 'failure_events', 'recovered_events'):
        row[metric + '_gross_parent'] = int(old[metric])
        row[metric + '_selected'] = int(current[metric])
        row[metric + '_delta'] = int(current[metric]) - int(old[metric])
    assert row['valid_tracking_frames_delta'] == 0
    rows.append(row)
with (dataset / 'gross_parent_paired_report/overall.csv').open(encoding='utf-8-sig') as stream:
    overall = {r['variant']: r for r in csv.DictReader(stream)}
summary = {}
for metric in ('MPR', 'MSR'):
    deltas = [r[metric + '_delta_pp'] for r in rows]
    expected = float(overall['selected_ABC'][metric]) - float(overall['gross_parent'][metric])
    assert math.isclose(math.fsum(deltas) / 234, expected, abs_tol=1e-9)
    negative = sorted((r for r in rows if r[metric + '_delta_pp'] < 0), key=lambda r: r[metric + '_delta_pp'])
    positive = sorted((r for r in rows if r[metric + '_delta_pp'] > 0), key=lambda r: r[metric + '_delta_pp'], reverse=True)
    negative_mass = -math.fsum(r[metric + '_delta_pp'] for r in negative)
    positive_mass = math.fsum(r[metric + '_delta_pp'] for r in positive)
    summary[metric] = dict(gross_parent_percent=float(overall['gross_parent'][metric]),
        selected_percent=float(overall['selected_ABC'][metric]), delta_pp=expected,
        improved_sequences=len(positive), worsened_sequences=len(negative), tied_sequences=sum(d == 0 for d in deltas),
        positive_sequence_pp_sum=positive_mass, negative_sequence_pp_sum=negative_mass,
        top5_negative_mass_fraction=-math.fsum(r[metric + '_delta_pp'] for r in negative[:5]) / negative_mass,
        largest_losses=negative[:10], largest_gains=positive[:10])

events = {}
for label in labels:
    records = core['variants'][label]['failure_event_records']
    censored = [r for r in records if not r['recovered']]
    assert len(records) == int(overall[label]['failure_events'])
    assert len(records) - len(censored) == int(overall[label]['recovered_events'])
    observed = [dict(sequence=r['sequence'], start_frame_zero_based=r['start_frame_zero_based'],
                     observed_nonrecovery_window_frames=lengths[r['sequence']] - 1 - r['start_frame_zero_based']) for r in censored]
    assert all(r['observed_nonrecovery_window_frames'] >= 0 for r in observed)
    events[label] = dict(failure_events=len(records), recovered_events=len(records) - len(censored),
        right_censored_events=len(censored), failed_frames=int(overall[label]['failed_tracking_frames']),
        conditional_recovered_delay_median=float(overall[label]['median_recovery_delay_frames']),
        censored_observed_nonrecovery_frame_sum=sum(r['observed_nonrecovery_window_frames'] for r in observed),
        censored_event_records=observed)

result = dict(completed=True, scope='Descriptive CPU arithmetic on existing full TEST tables and event records; not new testing or training/selection',
    sequences=234, frames=116649, metrics=summary, events=events,
    failed_frame_delta=events['selected_ABC']['failed_frames'] - events['gross_parent']['failed_frames'],
    labels_refer_to='Current aggregation selected epoch0 exactly reproduces prior trained relation epoch5; gross is old4/gross parent policy',
    limitations=['Different complete policy trajectories; neither event counts nor sequence correlations establish same-state causality.',
                 'Unrecovered means not recovered within this observed sequence, not permanently unrecoverable; observed windows are not all IoU<.2 frames.',
                 'Conditional recovered-event delay median excludes right-censored events and is not alone evidence of improved recovery.',
                 'These TEST data were already used in project development; not independent confirmation. No current running recipe or selection changed.'],
    next_check='Keep the current matched bidirectional experiment locked; compare its complete sequence/failure persistence against these descriptive losses after closure.')
out.mkdir()
with (out / 'per_sequence.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
(out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('sequences', 'frames', 'failed_frame_delta')}, ensure_ascii=False))
for metric, value in summary.items():
    print(metric, json.dumps({k: value[k] for k in ('delta_pp', 'improved_sequences', 'worsened_sequences', 'tied_sequences', 'top5_negative_mass_fraction')}))
    print('worst', [(r['sequence'], r[metric + '_delta_pp'], r['failed_tracking_frames_delta']) for r in value['largest_losses'][:5]])
print('events', json.dumps({label: {k: v for k, v in value.items() if k != 'censored_event_records'} for label, value in events.items()}))
