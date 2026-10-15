"""Independent CPU-only reproduction of the sealed RGBT210 pair; no neural imports."""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import pickletools
import sys
import zipfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / 'pure210_pair_audit_inputs_1230'
checks = []
TOL_PERCENT = 1e-10
TOL_CURVE = 1e-12
TOL_SECONDS = 1e-9


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, condition, details=None):
    checks.append(dict(name=name, status='PASS' if bool(condition) else 'FAIL', details=details))


def close(name, actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None
    check(name, error is not None and error <= tolerance,
          dict(shape=list(actual.shape), expected_shape=list(expected.shape), max_absolute_error=error, absolute_tolerance=tolerance))


def write_csv(name, columns, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def independent_curves(gt, raw, thresholds):
    pred = raw.round(0)
    pred[0] = gt[0]
    pc = (pred[:, 2:] - 1) / 2. + pred[:, :2]
    gc = (gt[:, 2:] - 1) / 2. + gt[:, :2]
    delta = pc - gc
    cle = (delta[:, 0] ** 2 + delta[:, 1] ** 2) ** .5
    pe = pred[:, 2:] + pred[:, :2] - 1
    ge = gt[:, 2:] + gt[:, :2] - 1
    overlap = np.maximum(0, np.minimum(pe, ge) - np.maximum(pred[:, :2], gt[:, :2]) + 1)
    inter = overlap[:, 0] * overlap[:, 1]
    union = pred[:, -1] * pred[:, -2] + gt[:, -1] * gt[:, -2] - inter
    iou = inter / union
    curves = {'PR': np.array([np.count_nonzero(cle <= t) / len(gt) for t in thresholds['PR']]),
              'SR': np.array([np.count_nonzero(iou > t) / len(gt) for t in thresholds['SR']])}
    diagnostics = dict(nonfinite_cle=int((~np.isfinite(cle)).sum()), nonfinite_iou=int((~np.isfinite(iou)).sum()),
        nonpositive_union=int((union <= 0).sum()), pr20_exact_boundary=int((cle == 20).sum()),
        sr_threshold_exact_equalities=[int((iou == t).sum()) for t in thresholds['SR']])
    return curves, diagnostics


def source_metric_namespace():
    # Only these already inspected pure numeric definitions are executed.
    namespace = {'np': np, '_strict': True, 'BaseRGBTDataet': object, 'TrackerResult': object,
                 'Metric': type('Metric', (), {})}
    sources = [('toolkit/rgbt/utils.py', {'ltwh_2_ltrb', 'serial_process', 'CLE', 'IoU'}),
               ('toolkit/rgbt/metrics/metrics.py', {'PR', 'SR'})]
    for relative, wanted in sources:
        parsed = ast.parse((ROOT / relative).read_text(encoding='utf-8'), filename=relative)
        nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
        assert {node.name for node in nodes} == wanted
        exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), namespace)
    return namespace


runtime = dict(python=sys.version, executable=sys.executable, numpy=np.__version__, numpy_file=np.__file__,
               dont_write_bytecode=sys.dont_write_bytecode, platform=sys.platform,
               generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write_json('runtime.json', runtime)
check('requested_runtime_python_numpy', sys.version_info[:3] == (3, 10, 19) and np.__version__ == '2.2.6')
check('python_bytecode_disabled', sys.dont_write_bytecode)

before = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
write_json('input_hashes_before.json', before)
sealed = json.loads((OUT / 'initial_input_hashes.json').read_text(encoding='utf-8'))
check('initial_input_seal_complete_equal', before == sealed,
      dict(actual_files=len(before), sealed_files=len(sealed), differences=[k for k in sorted(before.keys() | sealed.keys()) if before.get(k) != sealed.get(k)]))
manifest = read_json('raw_input_manifest.json')
manifest_rows = manifest['files']
manifest_bad = [row['file'] for row in manifest_rows if before.get(row['file']) != row['sha256'] or (ROOT / row['file']).stat().st_size != row['bytes']]
check('entire_collection_manifest_hashes_and_sizes', not manifest_bad and len({r['file'] for r in manifest_rows}) == len(manifest_rows),
      dict(manifest_entries=len(manifest_rows), mismatches=manifest_bad,
           additional_sealed_files=sorted(set(before) - {r['file'] for r in manifest_rows})))

protocols = read_json('toolkit_protocols.json')
protocol = protocols['rgbt210']
names = protocol['sequences']
thresholds = {key: np.array(value) for key, value in protocol['thresholds'].items()}
check('sequence_names_unique_210', len(names) == 210 and len(set(names)) == 210)
check('GT_sequence_coverage', {p.name for p in (ROOT / 'ground_truth/rgbt210').iterdir() if p.is_dir()} == set(names))
check('PR_native_threshold_grid', np.array_equal(thresholds['PR'], np.linspace(0, 50, 51)))
check('SR_native_threshold_grid', np.array_equal(thresholds['SR'], np.linspace(0, 1, 21)))
attr_root = ROOT / 'toolkit_attributes/rgbt210'
attr_names = (attr_root / 'SequencesName.txt').read_text(encoding='utf-8').split('\n')
attributes = {}
attribute_counts = {}
for attribute in ('BC', 'CM', 'DEF', 'FM', 'HO', 'LI', 'LR', 'MB', 'NO', 'TC', 'PO', 'SC'):
    flags = np.loadtxt(attr_root / (attribute + '.txt'), dtype=np.float32)
    members = [n for flag, n in zip(flags, attr_names[:len(flags)]) if flag and n in names]
    attributes[attribute] = members
    attribute_counts[attribute] = dict(original_flags=len(flags), selected_sequences=len(members))
    check('attribute_membership_' + attribute, members == protocol['attributes'][attribute] and len(set(members)) == len(members))
check('attribute_keys', set(attributes) == set(protocol['attributes']))
write_json('protocol_verified.json', dict(names=names, thresholds=protocol['thresholds'], attributes=attributes,
    attribute_counts=attribute_counts, input_manifest_metadata={k: v for k, v in manifest.items() if k != 'files'},
    metric_rule='GT and predictions float32; predictions round(0), then first prediction replaced with first GT; integer-inclusive ltwh IoU; strict centers; PR <= threshold; SR > threshold; all original frames kept; equal sequence weighting; SR arithmetic mean of 21 threshold success rates.'))

source_inventory = []
for path in sorted(ROOT.rglob('*.py')):
    source = path.read_text(encoding='utf-8')
    parsed = ast.parse(source, filename=str(path))
    source_inventory.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=before[path.relative_to(ROOT).as_posix()],
        lines=len(source.splitlines()), functions=[dict(name=n.name, line=n.lineno) for n in ast.walk(parsed) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]))
write_json('source_ast_inventory.json', source_inventory)
check('all_supplied_python_sources_parse', True, dict(files=len(source_inventory)))

gt_data = {}
gt_rows = []
for name in names:
    gt = np.loadtxt(ROOT / 'ground_truth/rgbt210' / name / 'init.txt', delimiter=',', dtype=np.float32, ndmin=2)
    check('GT_shape_finite_' + name, gt.ndim == 2 and gt.shape[1] == 4 and np.isfinite(gt).all())
    gt_data[name] = gt
    gt_rows.append([name, len(gt), int((gt[:, 2:] <= 0).any(1).sum()), int((gt[:, :2] <= 0).any(1).sum()),
                    int((gt <= 0).any(1).sum())])
write_csv('ground_truth_inventory.csv', ['sequence', 'frames', 'nonpositive_width_or_height_rows', 'nonpositive_x_or_y_rows', 'any_nonpositive_coordinate_rows'], gt_rows)
total_frames = sum(len(gt) for gt in gt_data.values())
check('original_frame_denominator_104706', total_frames == 104706)
native = source_metric_namespace()

roles = {}
for role, pred_rel, result_rel in [('main', 'predictions/main/rgbt210', 'results/main/rgbt210'),
                                 ('pure', 'predictions/pure_rgbt210', 'results/pure_rgbt210')]:
    pred_dir = ROOT / pred_rel
    report = read_json(result_rel + '/full_report.json')
    receipts = [read_json(pred_rel + '/complete_shard' + str(shard) + '.json') for shard in (0, 1)]
    csv_records = list(csv.DictReader((ROOT / result_rel / 'per_sequence.csv').open(encoding='utf-8')))
    check(role + '_prediction_coverage', {p.stem for p in pred_dir.glob('*.txt')} == set(names))
    check(role + '_latency_coverage', {p.name[:-len('_latency.npy')] for p in pred_dir.glob('*_latency.npy')} == set(names))
    check(role + '_report_per_sequence_order', [x['sequence'] for x in report['per_sequence']] == names)
    check(role + '_csv_per_sequence_order', [x['sequence'] for x in csv_records] == names)
    check(role + '_report_completed_count', report['completed'] and report['dataset'] == 'rgbt210' and report['sequences'] == 210 and report['frames'] == total_frames)
    receipt_records = {r['sequence']: r for receipt in receipts for r in receipt['records']}
    check(role + '_receipt_unique_coverage', len(receipt_records) == sum(len(s['records']) for s in receipts) == 210 and set(receipt_records) == set(names))
    epoch = 10 if role == 'main' else 14
    policy = 'search_actions' if role == 'main' else 'pure_cad'
    for shard, receipt in enumerate(receipts):
        selected_names = sorted(names)[shard::2]
        check(role + '_receipt_shard_' + str(shard), receipt['completed'] and receipt['dataset'] == 'rgbt210' and receipt['shard'] == shard and receipt['shards'] == 2
              and receipt['sequences'] == len(receipt['records']) == 105 and [r['sequence'] for r in receipt['records']] == selected_names
              and receipt['frames'] == sum(len(gt_data[n]) for n in selected_names) and receipt['epoch'] == report['epoch'] == epoch
              and receipt['checkpoint'] == report['checkpoint'] and receipt['policy_family'] == policy and receipt['first_box_only_initialization'])
    curves = {metric: [] for metric in thresholds}
    pred_data, latencies = {}, []
    per_seq, curve_rows, per_seq_latency, diag_rows = [], [], [], []
    for index, name in enumerate(names):
        gt = gt_data[name]
        raw = np.loadtxt(pred_dir / (name + '.txt'), delimiter='\t', dtype=np.float32, ndmin=2)
        check(role + '_prediction_shape_finite_' + name, raw.shape == gt.shape and np.isfinite(raw).all())
        check(role + '_receipt_frame_count_' + name, receipt_records[name]['frames'] == len(gt))
        check(role + '_initialization_box_' + name, np.array_equal(raw[0], np.round(gt[0], 3)))
        measured, diagnostics = independent_curves(gt, raw, thresholds)
        pred_data[name] = list(raw.round(0))
        row = [name, len(gt)]
        for metric in ('PR', 'SR'):
            curve = measured[metric]
            curves[metric].append(curve)
            value = float(curve[20] if metric == 'PR' else curve.mean()) * 100
            row.append(value)
            close(role + '_json_' + metric + '_' + name, value, report['per_sequence'][index][metric], TOL_PERCENT)
            close(role + '_csv_' + metric + '_' + name, value, float(csv_records[index][metric]), TOL_PERCENT)
            curve_rows.extend([name, metric, float(t), float(v)] for t, v in zip(thresholds[metric], curve))
        per_seq.append(row)
        latency = np.load(pred_dir / (name + '_latency.npy'), allow_pickle=False)
        check(role + '_latency_count_finite_positive_' + name, latency.shape == (len(gt) - 1,) and np.isfinite(latency).all() and (latency > 0).all())
        close(role + '_receipt_seconds_' + name, float(latency.sum()), receipt_records[name]['seconds'], TOL_SECONDS)
        latencies.append(latency)
        per_seq_latency.append([name, len(latency), float(latency.sum()), float(latency.mean() * 1000), float(np.percentile(latency, 50) * 1000), float(np.percentile(latency, 95) * 1000)])
        diag_rows.append(dict(sequence=name, **diagnostics))
    curves = {metric: np.asarray(rows) for metric, rows in curves.items()}
    overall, attribute_scores, attribute_curve_rows = {}, {}, []
    dataset = {n: {'visible': list(gt_data[n]), 'infrared': list(gt_data[n])} for n in names}
    for metric in ('PR', 'SR'):
        native_value, native_curve = native[metric]()(dataset, pred_data, names)
        close(role + '_independent_vs_source_' + metric + '_all_sequence_curves', curves[metric], native_curve, 0.)
        overall[metric] = float(curves[metric].mean(0)[20] if metric == 'PR' else curves[metric].mean()) * 100
        close(role + '_overall_' + metric, overall[metric], report['overall_metrics_percent'][metric], TOL_PERCENT)
        close(role + '_reported_mean_curve_' + metric, curves[metric].mean(0), report['mean_curves'][metric]['values'], TOL_CURVE)
        close(role + '_reported_thresholds_' + metric, thresholds[metric], report['mean_curves'][metric]['thresholds'], 0.)
    for attribute, members in attributes.items():
        indices = [names.index(name) for name in members]
        attribute_scores[attribute] = {'sequences': len(indices)}
        check(role + '_attribute_count_' + attribute, report['attributes'][attribute]['sequences'] == len(indices))
        for metric in ('PR', 'SR'):
            selected = curves[metric][indices]
            value = float(selected.mean(0)[20] if metric == 'PR' else selected.mean()) * 100
            attribute_scores[attribute][metric] = value
            close(role + '_attribute_' + attribute + '_' + metric, value, report['attributes'][attribute][metric], TOL_PERCENT)
            attribute_curve_rows.extend([attribute, len(indices), metric, float(t), float(v)] for t, v in zip(thresholds[metric], selected.mean(0)))
    latency = np.concatenate(latencies)
    timing = dict(samples=len(latency), sum_seconds=float(latency.sum()), fps=len(latency) / float(latency.sum()),
        latency_mean_ms=float(latency.mean() * 1000), latency_p50_ms=float(np.percentile(latency, 50) * 1000),
        latency_p95_ms=float(np.percentile(latency, 95) * 1000), latency_min_ms=float(latency.min() * 1000),
        latency_max_ms=float(latency.max() * 1000), excluded_initial_frames=210,
        interpretation='Per-worker recorded frame latency aggregates, not wall-clock pipeline throughput or a matched speedup.')
    for metric in ('fps', 'latency_p50_ms', 'latency_p95_ms'):
        close(role + '_timing_' + metric, timing[metric], report['timing'][metric], 1e-9)
    write_csv(role + '_per_sequence.csv', ['sequence', 'frames', 'PR_percent', 'SR_percent'], per_seq)
    write_csv(role + '_per_sequence_curves.csv', ['sequence', 'metric', 'threshold', 'success_fraction'], curve_rows)
    write_csv(role + '_mean_curves.csv', ['metric', 'threshold', 'success_fraction'], [[metric, float(t), float(v)] for metric in ('PR', 'SR') for t, v in zip(thresholds[metric], curves[metric].mean(0))])
    write_csv(role + '_attribute_curves.csv', ['attribute', 'sequences', 'metric', 'threshold', 'success_fraction'], attribute_curve_rows)
    write_csv(role + '_attribute_metrics.csv', ['attribute', 'sequences', 'PR_percent', 'SR_percent'], [[a, s['sequences'], s['PR'], s['SR']] for a, s in attribute_scores.items()])
    write_csv(role + '_per_sequence_latency.csv', ['sequence', 'samples', 'sum_seconds', 'mean_ms', 'p50_ms', 'p95_ms'], per_seq_latency)
    write_json(role + '_numeric_diagnostics.json', diag_rows)
    roles[role] = dict(overall_metrics_percent=overall, attributes=attribute_scores, timing=timing,
                      checkpoint=report['checkpoint'], epoch=epoch, sequence_scores=np.array([r[2:] for r in per_seq]),
                      source_report=result_rel + '/full_report.json')

main_state, pure_state = read_json('main_state.json'), read_json('pure210_state.json')
selected, fit = read_json('main_selection.json'), read_json('main_fit_complete.json')
identity = read_json('pure_checkpoint_identity.json')
config = read_json('project/cadtrack_abc/config.json')
check('main_checkpoint_hash_matches_state', before['selected_checkpoint/best.pth'] == main_state['checkpoint_sha256'])
check('main_config_hash_matches_state', before['project/cadtrack_abc/config.json'] == main_state['configuration_sha256'])
for file, digest in main_state['source_sha256'].items():
    check('main_declared_source_hash_' + file, before['project/cadtrack_abc/' + file] == digest)
check('pure_collection_checkpoint_identity_receipt', identity['path'] == pure_state['checkpoint'] == roles['pure']['checkpoint'] == config['pretrained']
      and identity['sha256'] == config['pretrained_sha256'] and identity['collection_time_only'] and identity['original_checkpoint_bytes_not_collected'])
check('main_selected_checkpoint_identity', roles['main']['checkpoint'] == main_state['checkpoint'] == selected['best_checkpoint'] == selected['trained_best_checkpoint']
      and selected['selected']['epoch'] == selected['trained_selected']['epoch'] == main_state['selected_epoch'] == roles['main']['epoch'])
check('main_declared_selection_rule', max(selected['candidates'], key=lambda c: c['sequence_mean_iou']) == selected['selected'] and len(selected['candidates']) == 6)
check('main_fit_updates_consistent', fit['updates'] == ((fit['train_queries'] + 63) // 64) * fit['epochs'] == 1600
      and fit['epochs'] == 20 and fit['policy_family'] == 'search_actions' and fit['training_hyperparameters']['seed'] == config['seed'] == 42)
with zipfile.ZipFile(ROOT / 'selected_checkpoint/best.pth') as archive:
    data_names = [name for name in archive.namelist() if name.endswith('/data.pkl')]
    assert len(data_names) == 1
    operations = list(pickletools.genops(archive.read(data_names[0])))
epoch_positions = [i for i, (op, argument, pos) in enumerate(operations) if op.name == 'BINUNICODE' and argument == 'epoch']
assert len(epoch_positions) == 1
epoch_tail = operations[epoch_positions[0] + 1:epoch_positions[0] + 4]
assert epoch_tail[0][0].name in ('BINPUT', 'LONG_BINPUT') and epoch_tail[1][0].name == 'BININT1'
payload_epoch = epoch_tail[1][1]
check('supplied_main_checkpoint_epoch_literal', payload_epoch == roles['main']['epoch'])
checkpoint_metadata = dict(bytes=(ROOT / 'selected_checkpoint/best.pth').stat().st_size, sha256=before['selected_checkpoint/best.pth'],
    epoch_literal=payload_epoch, inspection='zip and pickletools only; not unpickled or loaded into a neural model',
    pickle_globals=sorted({argument for op, argument, pos in operations if op.name == 'GLOBAL'}),
    pure_checkpoint_receipt=identity, historical_loaded_parameters_attested=False, historical_import_resolution_attested=False)
write_json('checkpoint_identity_verification.json', checkpoint_metadata)
check('main_recorded_completed_rgbt210_children', main_state['phases']['infer_rgbt210']['return_codes'] == [0, 0]
      and main_state['phases']['score_rgbt210']['return_codes'] == [0] and main_state['phases']['infer_rgbt210']['status'] == main_state['phases']['score_rgbt210']['status'] == 'VERIFIED_COMPLETE')
check('pure_recorded_completed_children', pure_state['status'] == 'PURE_RGBT210_COMPLETE' and pure_state['completed']
      and pure_state['inference_exit_codes'] == [0, 0] and pure_state['score_exit_code'] == 0)
observation = read_json('pure210_observation_1225.json')
check('pure_observation_state_matches', observation['state'] == pure_state and not observation['controller_actual_pid_live'])
stdout_lines = (ROOT / 'documentation/run_pure_rgbt210_gpu2_remote.stdout.txt').read_text(encoding='utf-8').splitlines()
check('pure_controller_stdout_final_state', json.loads(stdout_lines[-1]) == pure_state)
check('pure_controller_original_stderr_empty', (ROOT / 'documentation/run_pure_rgbt210_gpu2_remote.stderr.txt').stat().st_size == 0)
for role in ('main', 'pure'):
    expected_metrics = main_state['reports']['rgbt210']['metrics'] if role == 'main' else pure_state['metrics']
    for metric in ('PR', 'SR'):
        close(role + '_controller_report_' + metric, roles[role]['overall_metrics_percent'][metric], expected_metrics[metric], TOL_PERCENT)

differences = roles['main']['sequence_scores'] - roles['pure']['sequence_scores']
rng = np.random.default_rng(20261015)
indices = rng.integers(0, len(names), size=(5000, len(names)))
replicates = differences[indices].mean(1)
intervals = np.quantile(replicates, [.025, .975], axis=0, method='linear')
bootstrap = dict(method='Paired sequence bootstrap; shared named-sequence draws for PR and SR', seed=20261015,
    generator='numpy.random.default_rng / PCG64', bit_generator=type(rng.bit_generator).__name__, replicates=5000,
    resampled_sequences=210, sampling='with replacement, 210 equally weighted named sequences per replicate',
    sequence_order=names, quantiles=[.025, .975], quantile_method='linear', units='percentage points, main minus pure',
    index_shape=list(indices.shape), index_sha256_int64_little_endian=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
    metrics={metric: dict(point_estimate=float(differences[:, j].mean()), percentile95=[float(x) for x in intervals[:, j]],
        positive_sequences=int((differences[:, j] > 0).sum()), equal_sequences=int((differences[:, j] == 0).sum()), negative_sequences=int((differences[:, j] < 0).sum()),
        bootstrap_positive_fraction=float((replicates[:, j] > 0).mean())) for j, metric in enumerate(('PR', 'SR'))},
    limits=['Fixed selected checkpoints and supplied benchmark sequences only; no retraining or training-seed inference.',
            'Within-video frames are dependent, so the resampling unit is a full sequence.',
            'Related events or captures across named sequences may remain dependent; these intervals assume exchangeable sequences.',
            'Repeated development selection and historical test exposure are not incorporated; no causal module-effect or generalization guarantee.',
            'Two marginal 95% percentile intervals, not a simultaneous confidence region; no multiplicity adjustment.'])
write_json('paired_bootstrap_5000.json', bootstrap)
write_csv('paired_bootstrap_5000_replicates.csv', ['replicate', 'delta_PR_pp', 'delta_SR_pp'], [[i + 1, *row] for i, row in enumerate(replicates.tolist())])
write_csv('paired_sequence_differences.csv', ['sequence', 'main_PR_percent', 'pure_PR_percent', 'delta_PR_pp', 'main_SR_percent', 'pure_SR_percent', 'delta_SR_pp'],
          [[name, roles['main']['sequence_scores'][i, 0], roles['pure']['sequence_scores'][i, 0], differences[i, 0],
            roles['main']['sequence_scores'][i, 1], roles['pure']['sequence_scores'][i, 1], differences[i, 1]] for i, name in enumerate(names)])
write_json('aggregate_latency_statistics.json', {role: record['timing'] for role, record in roles.items()})
for record in roles.values():
    del record['sequence_scores']
write_json('recomputed_results.json', dict(sequences=210, frames_per_role=total_frames, roles=roles,
    gt_nonpositive_width_height_rows=sum(row[2] for row in gt_rows), gt_any_nonpositive_coordinate_rows=sum(row[4] for row in gt_rows)))
after = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
write_json('input_hashes_after.json', after)
check('input_files_unchanged_during_verification', before == after)
verification = dict(status='PASS' if all(c['status'] == 'PASS' for c in checks) else 'FAIL',
    generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), runtime=runtime,
    absolute_tolerances=dict(metric_percent=TOL_PERCENT, curve=TOL_CURVE, latency_seconds=TOL_SECONDS, source_curve_crosscheck=0.),
    input_files=len(before), collection_manifest_entries=len(manifest_rows), source_python_files=len(source_inventory),
    total_checks=len(checks), failed_checks=[c for c in checks if c['status'] == 'FAIL'], checks=checks)
write_json('deterministic_verification.json', verification)
print(json.dumps(dict(status=verification['status'], checks=len(checks), failed=verification['failed_checks'],
    input_files=len(before), frames_per_role=total_frames, metrics={r: data['overall_metrics_percent'] for r, data in roles.items()},
    paired_bootstrap=bootstrap['metrics'], invalid_gt_rows=sum(row[2] for row in gt_rows)), indent=2))
raise SystemExit(0 if verification['status'] == 'PASS' else 1)
