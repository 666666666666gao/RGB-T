"""CPU-only audit. Inputs are read only; only audit outputs are written.

Replays the supplied metric class bodies (imports removed to avoid plotting
and absent dataset files), then cross-checks an independent NumPy formula.
No torch, checkpoint loading, images, model inference, or network operations.
"""
import ast
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_inputs_1040')
OUT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_lasher_formal_audit_20261015')
PATHS = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_paths.json')

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

class Checks:
    def __init__(self):
        self.groups = {}
        self.failures = []

    def check(self, group, label, condition):
        record = self.groups.setdefault(group, {'comparisons': 0, 'failed': 0})
        record['comparisons'] += 1
        if not bool(condition):
            record['failed'] += 1
            self.failures.append({'group': group, 'label': label})

    def close(self, group, label, actual, expected, atol=1e-12):
        a, e = np.asarray(actual), np.asarray(expected)
        self.check(group, label + ':shape', a.shape == e.shape)
        if a.shape != e.shape:
            return
        errors = np.abs(a - e)
        passed = np.isfinite(errors) & (errors <= atol)
        record = self.groups[group]
        record['comparisons'] += int(passed.size)
        record['failed'] += int((~passed).sum())
        record['max_abs_error'] = max(record.get('max_abs_error', 0.), float(errors.max(initial=0.)))
        if not passed.all():
            self.failures.append({'group': group, 'label': label,
                                  'failed_values': int((~passed).sum()),
                                  'max_abs_error': float(errors.max())})

started = time.perf_counter()
checks = Checks()
paths = load(PATHS)['files']
before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
for p, digest in load(OUT / 'initial_input_hashes.json').items():
    checks.check('input_before_hashes', str(Path(p).relative_to(ROOT)), before[p] == digest)
raw_manifest = load(ROOT / 'raw_input_manifest.json')
source_rows = {r['file']: r for r in raw_manifest['files']}
for rel, record in source_rows.items():
    path = ROOT / rel
    checks.check('source_provenance', rel + ':hash', hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'])
    checks.check('source_provenance', rel + ':bytes', path.stat().st_size == record['bytes'])

report = load(ROOT / 'report/full_report.json')
protocol = load(ROOT / 'toolkit_protocol.json')
names = protocol['sequences']
metrics = ['NPR', 'PR', 'SR']
thresholds = {'NPR': np.linspace(0, .5, 51), 'PR': np.linspace(0, 50, 51), 'SR': np.linspace(0, 1, 21)}
expected_names = set(names)
checks.check('cohort', 'protocol_count_unique', len(names) == len(expected_names) == 245)
checks.check('cohort', 'gt_names_exact', {p.name for p in (ROOT / 'ground_truth').iterdir()} == expected_names)
checks.check('cohort', 'pred_names_exact', {p.stem for p in (ROOT / 'predictions').glob('*.txt')} == expected_names)
checks.check('cohort', 'latency_names_exact', {p.name[:-12] for p in (ROOT / 'predictions').glob('*_latency.npy')} == expected_names)
checks.check('cohort', 'report_names_exact_order', [r['sequence'] for r in report['per_sequence']] == names)
checks.check('cohort', 'attribute_names_exact', list(report['attributes']) == list(protocol['attributes']))
for metric in metrics:
    checks.close('thresholds', metric + ':export', protocol['thresholds'][metric], thresholds[metric], 0.)
    checks.close('thresholds', metric + ':report', report['mean_curves'][metric]['thresholds'], thresholds[metric], 0.)

receipts = [load(ROOT / f'predictions/complete_shard{s}.json') for s in (0, 1)]
records = {}
for shard, receipt in enumerate(receipts):
    checks.check('receipts', f'shard{shard}:terminal_identity', receipt['completed'] and receipt['dataset'] == 'lasher' and receipt['shard'] == shard and receipt['shards'] == 2)
    checks.check('receipts', f'shard{shard}:checkpoint', receipt['checkpoint'] == report['checkpoint'] and receipt['epoch'] == report['epoch'] == 10 and receipt['policy_family'] == 'search_actions')
    checks.check('receipts', f'shard{shard}:init_load_flags', receipt['first_box_only_initialization'] and receipt['strict_base_and_abc_load'])
    checks.check('receipts', f'shard{shard}:sorted_partition', [r['sequence'] for r in receipt['records']] == sorted(names)[shard::2])
    checks.check('receipts', f'shard{shard}:sequence_count', len(receipt['records']) == receipt['sequences'])
    checks.check('receipts', f'shard{shard}:frame_total', sum(r['frames'] for r in receipt['records']) == receipt['frames'])
    for row in receipt['records']:
        checks.check('receipts', row['sequence'] + ':unique', row['sequence'] not in records)
        records[row['sequence']] = row

gt, predictions, latencies, private_stats = {}, {}, {}, []
source_gt_paths_correct = 0
for name in names:
    g = np.loadtxt(ROOT / 'ground_truth' / name / 'init.txt', delimiter=',', ndmin=2)
    p = np.loadtxt(ROOT / 'predictions' / (name + '.txt'), ndmin=2)
    latency = np.load(ROOT / 'predictions' / (name + '_latency.npy'), allow_pickle=False)
    gt[name], predictions[name], latencies[name] = g, p, latency
    checks.check('raw_shapes', name + ':finite_gt', g.ndim == 2 and g.shape[1] == 4 and np.isfinite(g).all())
    checks.check('raw_shapes', name + ':finite_prediction_shape', p.shape == g.shape and np.isfinite(p).all())
    checks.check('raw_shapes', name + ':latency_shape_finite_positive', latency.shape == (len(g) - 1,) and np.isfinite(latency).all() and (latency > 0).all())
    checks.check('raw_shapes', name + ':receipt_frames', len(g) == records[name]['frames'])
    checks.close('latency_receipts', name, float(latency.sum()), records[name]['seconds'], 1e-9)
    expected_first = np.asarray([float('%.3f' % v) for v in g[0].astype(np.float32)])
    checks.close('initialization', name + ':serialized_first_box', p[0], expected_first, 0.)
    checks.check('initialization', name + ':positive_initial_dimensions', (g[0, 2:] > 0).all())
    source_path = source_rows[f'ground_truth/{name}/init.txt']['source']
    source_gt_paths_correct += source_path == raw_manifest['actual_dataset_root'] + '/' + name + '/init.txt'
    checks.check('gt_source_paths', name, source_path == report['ground_truth_root'] + '/' + name + '/init.txt')
    rounded = p.astype(np.float32).round(0)
    invalid = (g <= 0).any(axis=1)
    private_stats.append({'sequence': name, 'frames': len(g), 'tracked_frames': len(g) - 1,
                          'official_invalid_gt_frames': int(invalid.sum()),
                          'nonpositive_gt_dimensions': int((g[:, 2:] <= 0).any(axis=1).sum()),
                          'invalid_gt_xy_only': int((invalid & (g[:, 2:] > 0).all(axis=1)).sum()),
                          'prediction_dimension_repair_frames': int((rounded[1:, 2:] <= 0).any(axis=1).sum()),
                          'prediction_values_changed_by_rounding': int((p != rounded).sum()),
                          'noninitial_exact_gt_prediction_frames': int((p[1:] == g[1:]).all(axis=1).sum()),
                          'gt_pred_byte_hash_equal': hashlib.sha256((ROOT / 'ground_truth' / name / 'init.txt').read_bytes()).digest() == hashlib.sha256((ROOT / 'predictions' / (name + '.txt')).read_bytes()).digest()})

frame_count = sum(len(g) for g in gt.values())
checks.check('cohort', 'total_frames', frame_count == report['frames'] == 220703)
checks.check('cohort', 'total_sequences', report['sequences'] == len(names))
checks.check('cohort', 'completed_lasher', report['completed'] and report['dataset'] == 'lasher')

# Execute the original supplied mathematical bodies. Do not import rgbt, whose
# package initializer imports plotting and whose default data are not supplied.
toolkit_env = {'__name__': 'audit_toolkit_replay', 'BaseRGBTDataet': dict, 'TrackerResult': dict, 'os': os}
exec(compile((ROOT / 'toolkit/rgbt/utils.py').read_text(encoding='utf-8'), str(ROOT / 'toolkit/rgbt/utils.py'), 'exec'), toolkit_env)
for rel in ('toolkit/rgbt/metrics/base.py', 'toolkit/rgbt/metrics/metrics.py'):
    parsed = ast.parse((ROOT / rel).read_text(encoding='utf-8'), filename=rel)
    parsed.body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(parsed, str(ROOT / rel), 'exec'), toolkit_env)
parsed = ast.parse((ROOT / 'toolkit/rgbt/dataset/basedataset.py').read_text(encoding='utf-8'))
parsed.body = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'initial_result_file']
exec(compile(parsed, str(ROOT / 'toolkit/rgbt/dataset/basedataset.py'), 'exec'), toolkit_env)
loaded = toolkit_env['initial_result_file'](str(ROOT / 'predictions'), names, lambda box: box)
toolkit_curves, toolkit_values = {}, {}
class_names = {'NPR': 'NPR', 'PR': 'PR_LasHeR', 'SR': 'SR_LasHeR'}
for metric in metrics:
    function = toolkit_env[class_names[metric]]()
    checks.close('thresholds', metric + ':class', function.thr, thresholds[metric], 0.)
    value, curve = function(gt, loaded, names)
    toolkit_curves[metric], toolkit_values[metric] = curve, float(value) * 100
    print(json.dumps({'replayed': metric, 'score_percent': toolkit_values[metric]}), flush=True)

# Independently reproduce definitions, preserving float32 integer rounding,
# first-GT insertion, invalid-dimension carry forward, and invalid-GT sentinels.
independent = {metric: [] for metric in metrics}
invalid_pr_npr_bonus = []
for name in names:
    truth = gt[name]
    pred = predictions[name].astype(np.float32).round(0).astype(np.float64)
    pred[0] = truth[0]
    for i in range(1, len(pred)):
        if (pred[i, 2:] <= 0).any():
            pred[i] = pred[i - 1]
    pred_center = pred[:, :2] + (pred[:, 2:] - 1) / 2
    truth_center = truth[:, :2] + (truth[:, 2:] - 1) / 2
    center_error = np.sqrt(np.square(pred_center - truth_center).sum(axis=1))
    norm_error = np.sqrt(np.square(pred_center / (truth[:, 2:] + 1e-8) - truth_center / (truth[:, 2:] + 1e-8)).sum(axis=1))
    near = np.maximum(pred[:, :2], truth[:, :2])
    far = np.minimum(pred[:, :2] + pred[:, 2:] - 1, truth[:, :2] + truth[:, 2:] - 1)
    intersection = np.maximum(far - near + 1, 0).prod(axis=1)
    overlap = intersection / (pred[:, 2:].prod(axis=1) + truth[:, 2:].prod(axis=1) - intersection)
    invalid = (truth <= 0).any(axis=1)
    center_error[invalid] = norm_error[invalid] = overlap[invalid] = -1
    independent['NPR'].append((norm_error[:, None] <= thresholds['NPR']).mean(axis=0))
    independent['PR'].append((center_error[:, None] <= thresholds['PR']).mean(axis=0))
    independent['SR'].append((overlap[:, None] > thresholds['SR']).mean(axis=0))
    invalid_pr_npr_bonus.append(float(invalid.mean() * 100))
for metric in metrics:
    independent[metric] = np.asarray(independent[metric])
    checks.close('independent_vs_toolkit_curves', metric, independent[metric], toolkit_curves[metric], 0.)
    checks.close('reported_overall', metric, toolkit_values[metric], report['overall_metrics_percent'][metric])
    checks.close('reported_mean_curves', metric, toolkit_curves[metric].mean(axis=0), report['mean_curves'][metric]['values'])

with (ROOT / 'report/per_sequence.csv').open(encoding='utf-8', newline='') as stream:
    csv_rows = list(csv.DictReader(stream))
checks.check('cohort', 'csv_names_exact_order', [r['sequence'] for r in csv_rows] == names)
recomputed_sequences, curve_rows = [], []
for i, name in enumerate(names):
    row = dict(private_stats[i])
    for metric in metrics:
        curve = toolkit_curves[metric][i]
        value = float(curve.mean() if metric == 'SR' else curve[20]) * 100
        row[metric + '_percent'] = value
        checks.close('reported_per_sequence_json', name + ':' + metric, value, report['per_sequence'][i][metric])
        checks.close('reported_per_sequence_csv', name + ':' + metric, value, float(csv_rows[i][metric]))
        for threshold, fraction in zip(thresholds[metric], curve):
            curve_rows.append({'sequence': name, 'metric': metric, 'threshold': float(threshold), 'fraction': float(fraction)})
    recomputed_sequences.append(row)
write_csv('verified_per_sequence.csv', recomputed_sequences)
write_csv('verified_per_sequence_curves.csv', curve_rows)
attributes = []
for attribute, members in protocol['attributes'].items():
    checks.check('attributes', attribute + ':unique_members_subset', len(members) == len(set(members)) and set(members) <= expected_names)
    checks.check('attributes', attribute + ':sequence_count', len(members) == report['attributes'][attribute]['sequences'])
    indices = [names.index(name) for name in members]
    row = {'attribute': attribute, 'sequences': len(indices)}
    for metric in metrics:
        curves = toolkit_curves[metric][indices]
        value = float(curves.mean() if metric == 'SR' else curves.mean(axis=0)[20]) * 100
        row[metric + '_percent'] = value
        checks.close('reported_attributes', attribute + ':' + metric, value, report['attributes'][attribute][metric])
    attributes.append(row)
write_csv('verified_attributes.csv', attributes)

latency = np.concatenate([latencies[name] for name in names])
timing = {'tracked_frames': int(len(latency)), 'sum_worker_frame_seconds': float(latency.sum()),
          'mean_frame_latency_ms': float(latency.mean() * 1000),
          'fps_reciprocal_mean_frame_latency': len(latency) / float(latency.sum()),
          'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
          'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
          'wall_clock_throughput_verified': False,
          'matched_baseline_cost_verified': False}
checks.check('latency', 'count_excludes_all_first_frames', len(latency) == frame_count - len(names))
for report_key, value in [('fps', timing['fps_reciprocal_mean_frame_latency']), ('latency_p50_ms', timing['latency_p50_ms']), ('latency_p95_ms', timing['latency_p95_ms'])]:
    checks.close('reported_latency', report_key, value, report['timing'][report_key])

selection = load(ROOT / 'selection_snapshot.json')
fit = load(ROOT / 'fit_complete.json')
observation = load(ROOT / 'main_observation_1040.json')
selected = max(selection['candidates'], key=lambda r: r['sequence_mean_iou'])
trained = max((r for r in selection['candidates'] if r['epoch'] > 0), key=lambda r: r['sequence_mean_iou'])
checks.check('selection', 'best_of_recorded_six', selection['selected'] == selected and selected['epoch'] == 10)
checks.check('selection', 'trained_best_of_recorded_six', selection['trained_selected'] == trained and trained == selected)
checks.check('selection', 'same_checkpoint_path', selection['best_checkpoint'] == selection['trained_best_checkpoint'] == report['checkpoint'] == observation['checkpoint'])
checks.check('selection', 'fit_candidate_schedule', [(r['epoch'], r['checkpoint']) for r in fit['candidates']] == [(r['epoch'], r['checkpoint']) for r in selection['candidates']])
checks.check('selection', 'fit_update_arithmetic', math.ceil(fit['train_queries'] / fit['training_hyperparameters']['batch']) * fit['epochs'] == fit['updates'] == 1600)
checks.check('selection', 'single_seed_recorded', fit['training_hyperparameters']['seed'] == 42)
checks.check('selection', 'full_dev_flag', selection['full98_development_selection_complete'])
for metric in metrics:
    checks.close('observation_report', metric, observation['reports']['lasher']['metrics'][metric], report['overall_metrics_percent'][metric])
    ref = report['baseline_metrics_percent_user_provided'][metric]
    checks.close('claim_arithmetic', metric + ':delta', toolkit_values[metric] - ref, report['delta_percentage_points_vs_user_reference'][metric])
    target = ref + report['lasher_stage_goal']['required_gain_percentage_points']
    checks.close('claim_arithmetic', metric + ':target', target, report['lasher_stage_goal']['targets_percent'][metric])
    checks.check('claim_arithmetic', metric + ':target_reached', report['lasher_stage_goal']['per_metric_reached'][metric] == (toolkit_values[metric] >= target))
checks.check('claim_arithmetic', 'all_target_reached', report['lasher_stage_goal']['all_three_reached'] == all(report['lasher_stage_goal']['per_metric_reached'].values()))

# Mechanism metrics that can be reconstructed from GT and final predictions.
# Candidate sets / commits / keep-box traces are not supplied and are not
# silently treated as verified by this partial reconstruction.
mechanism = load(ROOT / 'report/mechanism_report.json')
all_events, mechanism_rows, total_iou, valid_frames, failure_frames = [], [], 0., 0, 0
for name in names:
    truth, pred = gt[name][1:], predictions[name][1:]
    valid = (truth[:, 2:] > 0).all(axis=1)
    near = np.maximum(pred[:, :2], truth[:, :2])
    far = np.minimum(pred[:, :2] + pred[:, 2:], truth[:, :2] + truth[:, 2:])
    inter = np.maximum(far - near, 0).prod(axis=1)
    iou = inter / np.maximum(np.maximum(pred[:, 2:], 0).prod(axis=1) + np.maximum(truth[:, 2:], 0).prod(axis=1) - inter, 1e-12)
    events, failure_start = [], None
    for frame, (value, is_valid) in enumerate(zip(iou, valid), 1):
        if not is_valid:
            continue
        if failure_start is None and value < .2:
            failure_start = frame
        elif failure_start is not None and value >= .5:
            events.append({'sequence': name, 'start_frame': failure_start, 'recovery_frame': frame, 'delay_frames': frame - failure_start, 'right_censored': False})
            failure_start = None
    if failure_start is not None:
        events.append({'sequence': name, 'start_frame': failure_start, 'recovery_frame': None, 'delay_frames': None, 'observed_until_frame': len(iou), 'right_censored': True})
    all_events.extend(events)
    mechanism_rows.append({'sequence': name, 'valid_gt_frames': int(valid.sum()), 'mean_iou': float(iou[valid].mean()), 'failure_events': len(events)})
    total_iou += float(iou[valid].sum())
    valid_frames += int(valid.sum())
    failure_frames += int((valid & (iou < .2)).sum())
checks.check('mechanism_gt_outputs', 'tracked_frames', mechanism['counts']['tracked_frames'] == len(latency))
checks.check('mechanism_gt_outputs', 'valid_frames', mechanism['counts']['valid_gt_frames'] == valid_frames)
checks.check('mechanism_gt_outputs', 'failure_frames', mechanism['counts']['failure_frames'] == failure_frames)
checks.check('mechanism_gt_outputs', 'all_events_length', len(all_events) == len(mechanism['events']) == mechanism['failure_events'])
for i, event in enumerate(all_events):
    checks.check('mechanism_gt_outputs', 'event:' + str(i), event == mechanism['events'][i])
for i, row in enumerate(mechanism_rows):
    actual = mechanism['per_sequence'][i]
    checks.check('mechanism_gt_outputs', row['sequence'] + ':counts', all(row[k] == actual[k] for k in ['sequence', 'valid_gt_frames', 'failure_events']))
    checks.close('mechanism_gt_outputs', row['sequence'] + ':mean_iou', row['mean_iou'], actual['mean_iou'])
recovered = [event['delay_frames'] for event in all_events if not event['right_censored']]
mechanism_verified = {'tracked_frames': len(latency), 'valid_gt_frames': valid_frames, 'failure_frames': failure_frames,
                      'sequence_equal_mean_iou': float(np.mean([r['mean_iou'] for r in mechanism_rows])),
                      'frame_equal_mean_iou': total_iou / valid_frames,
                      'failure_events': len(all_events), 'recovered_events': len(recovered),
                      'right_censored_events': len(all_events) - len(recovered),
                      'recovery_delay_median_frames': float(np.median(recovered))}
for key, value in mechanism_verified.items():
    if key not in ('tracked_frames', 'valid_gt_frames', 'failure_frames'):
        checks.close('mechanism_gt_outputs', key, value, mechanism[key])
ratio_inputs = {'candidate_recall': ('candidate_recalled_frames', 'valid_gt_frames'),
                'local_candidate_recall': ('local_candidate_recalled_frames', 'valid_gt_frames'),
                'newly_recalled_acceptance': ('newly_recalled_selected_frames', 'newly_recalled_frames'),
                'incorrect_template_commit_fraction': ('incorrect_template_commit_frames', 'template_commit_frames'),
                'incorrect_geometry_commit_fraction': ('incorrect_geometry_commit_frames', 'geometry_commit_frames'),
                'extra_forward_fraction': ('extra_visual_forwards', 'tracked_frames')}
for key, (numerator, denominator) in ratio_inputs.items():
    checks.close('mechanism_internal_arithmetic_only', key, mechanism['counts'][numerator] / mechanism['counts'][denominator], mechanism[key])
checks.check('mechanism_receipt_aggregation_only', 'extra_forwards_total', sum(r['extra_forwards'] for r in records.values()) == mechanism['counts']['extra_visual_forwards'])

after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
dump('input_hashes_after.json', after)
for p, digest in before.items():
    checks.check('input_after_hashes', str(Path(p).relative_to(ROOT)), after[p] == digest)
stats = {key: sum(int(r[key]) for r in private_stats) for key in private_stats[0] if key != 'sequence'}
stats['sequences_with_official_invalid_gt'] = sum(r['official_invalid_gt_frames'] > 0 for r in private_stats)
stats['sequence_equal_invalid_gt_precision_credit_pp'] = float(np.mean(invalid_pr_npr_bonus))
result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - started,
          'python_executable': sys.executable, 'python_version': sys.version, 'numpy_version': np.__version__,
          'numpy_file': np.__file__, 'total_comparisons': sum(r['comparisons'] for r in checks.groups.values()),
          'failed_comparisons': sum(r['failed'] for r in checks.groups.values()),
          'groups': checks.groups, 'failures': checks.failures,
          'overall_metrics_percent': toolkit_values, 'dataset_counts': stats,
          'timing': timing, 'verified_mechanism_subset': mechanism_verified,
          'selected_epoch_from_recorded_candidates': selected['epoch'],
          'checkpoint_hash_observation_only': observation['checkpoint_sha256'],
          'input_file_count': len(paths), 'source_manifest_file_count': len(source_rows),
          'source_manifest_does_not_cover': [str(Path(p).relative_to(ROOT)).replace('\\', '/') for p in paths if str(Path(p).relative_to(ROOT)).replace('\\', '/') not in source_rows],
          'artifacts': ['verified_per_sequence.csv', 'verified_per_sequence_curves.csv', 'verified_attributes.csv', 'input_hashes_before.json', 'input_hashes_after.json'],
          'scope': 'Completed LasHeR standard metrics are reproduced; no neural execution, upstream official-distribution certification, checkpoint-byte verification, full DEV recomputation, or unavailable per-frame mechanism claims.'}
dump('deterministic_verification.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result['failed_comparisons'] == 0 else 1)
