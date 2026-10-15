"""Validate final audit serialization and invariants without rerunning neural work."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / 'pure210_pair_audit_inputs_1230'
checks = []
def check(name, ok, detail=None):
    checks.append(dict(name=name, status='PASS' if bool(ok) else 'FAIL', detail=detail))
def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))
def csv_rows(name):
    with (OUT / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

audit = read('EXPERIMENT_AUDIT.json')
verification = read('deterministic_verification.json')
summary = read('recomputed_results.json')
bootstrap = read('paired_bootstrap_5000.json')
report = (OUT / 'EXPERIMENT_AUDIT.md').read_text(encoding='utf-8')
check('review_verdict_consistent', audit['verdict'] == 'WARN' and audit['overall_verdict'] == audit['integrity_status'] == 'warn' and not audit['blocking_issues'])
check('A_to_F_complete', len(audit['checks']) == 6 and {k[0] for k in audit['checks']} == set('ABCDEF'))
check('review_route_not_overclaimed', audit['reviewer_model'] == audit['reviewer_reasoning'] == audit['actual_routing'] == 'UNATTESTED'
      and audit['review_independence'] == 'same-family' and audit['acceptance_status'] == 'provisional')
check('deterministic_summary_agrees', verification['status'] == 'PASS' and verification['total_checks'] == len(verification['checks']) == audit['deterministic_verification']['named_checks'] == 4154
      and not verification['failed_checks'] and all(c['status'] == 'PASS' for c in verification['checks']))
for section in audit['checks'].values():
    for item in section['evidence']:
        path = Path(item['path'])
        check('evidence_' + path.name + '_' + str(item['line']), path.is_file() and 1 <= item['line'] <= len(path.read_text(encoding='utf-8').splitlines()))
trace = OUT / '.aris/traces/experiment-audit/2026-10-15_run01'
check('full_request_trace', json.loads((trace / '001-pair-audit.request.json').read_text(encoding='utf-8'))['prompt'] == (OUT / 'review_request.txt').read_text(encoding='utf-8'))
check('full_written_response_trace', (trace / '001-pair-audit.response.md').read_text(encoding='utf-8') == report)

names = bootstrap['sequence_order']
expected_counts = {'ground_truth_inventory.csv': 210, 'paired_sequence_differences.csv': 210, 'paired_bootstrap_5000_replicates.csv': 5000}
for role in ('main', 'pure'):
    expected_counts.update({role + '_per_sequence.csv': 210, role + '_per_sequence_curves.csv': 15120,
                           role + '_mean_curves.csv': 72, role + '_attribute_curves.csv': 864,
                           role + '_attribute_metrics.csv': 12, role + '_per_sequence_latency.csv': 210})
for name, count in expected_counts.items():
    rows = csv_rows(name)
    check('serialized_row_count_' + name, len(rows) == count, dict(expected=count, actual=len(rows)))
    check('no_raw_coordinate_columns_' + name, not ({'x', 'y', 'w', 'h', 'x1', 'y1', 'x2', 'y2', 'bbox', 'gt_box', 'prediction_box'} & set(rows[0])))
check('no_binary_coordinate_checkpoint_or_latency_outputs', not [p.name for p in OUT.rglob('*') if p.is_file() and p.suffix.lower() in ('.npy', '.npz', '.pth', '.tar', '.bin')])
for role in ('main', 'pure'):
    seq = csv_rows(role + '_per_sequence.csv')
    check(role + '_named_sequence_order_saved', [r['sequence'] for r in seq] == names)
    for metric in ('PR', 'SR'):
        mean = np.mean([float(r[metric + '_percent']) for r in seq])
        check(role + '_' + metric + '_serialized_mean', abs(mean - summary['roles'][role]['overall_metrics_percent'][metric]) <= 1e-10)
    curves = csv_rows(role + '_per_sequence_curves.csv')
    check(role + '_serialized_curves_finite_bounded', all(np.isfinite(float(r['success_fraction'])) and 0 <= float(r['success_fraction']) <= 1 for r in curves))
    for metric in ('PR', 'SR'):
        width = 51 if metric == 'PR' else 21
        values = np.array([float(row['success_fraction']) for row in curves if row['metric'] == metric]).reshape(210, width)
        check(role + '_' + metric + '_curve_monotonicity', (np.diff(values, axis=1) >= 0).all() if metric == 'PR' else (np.diff(values, axis=1) <= 0).all())
        extracted = values[:, 20] * 100 if metric == 'PR' else values.mean(1) * 100
        check(role + '_' + metric + '_per_sequence_matches_curve', np.array_equal(extracted, [float(row[metric + '_percent']) for row in seq]))
    latency_rows = csv_rows(role + '_per_sequence_latency.csv')
    check(role + '_latency_count_serialization', sum(int(row['samples']) for row in latency_rows) == summary['roles'][role]['timing']['samples'] == 104496)

deltas = csv_rows('paired_sequence_differences.csv')
check('paired_saved_sequence_order', [r['sequence'] for r in deltas] == names)
delta_array = np.array([[float(row['delta_PR_pp']), float(row['delta_SR_pp'])] for row in deltas])
rng = np.random.default_rng(20261015)
indices = rng.integers(0, len(names), size=(5000, len(names)))
regenerated = delta_array[indices].mean(1)
stored = np.array([[float(row['delta_PR_pp']), float(row['delta_SR_pp'])] for row in csv_rows('paired_bootstrap_5000_replicates.csv')])
check('paired_replicates_exactly_reproducible_from_saved_deltas', np.array_equal(regenerated, stored))
check('paired_rng_index_hash', hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest() == bootstrap['index_sha256_int64_little_endian'])
quantiles = np.quantile(stored, [.025, .975], axis=0, method='linear')
for index, metric in enumerate(('PR', 'SR')):
    check(metric + '_saved_percentile_endpoints_exact', np.array_equal(quantiles[:, index], bootstrap['metrics'][metric]['percentile95']))
    check(metric + '_point_estimate_exact', float(delta_array[:, index].mean()) == bootstrap['metrics'][metric]['point_estimate'])
    check(metric + '_interval_contains_zero', quantiles[0, index] < 0 < quantiles[1, index])

initial = read('initial_input_hashes.json')
final = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file()}
(OUT / 'input_hashes_final.json').write_text(json.dumps(final, indent=2) + '\n', encoding='utf-8')
check('final_input_set_and_all_sha256_unchanged', final == initial == read('input_hashes_before.json') == read('input_hashes_after.json'), dict(files=len(final)))
for path in sorted(OUT.rglob('*.json')):
    json.loads(path.read_text(encoding='utf-8'))
check('all_existing_json_artifacts_parse', True)
for relative in ('execution/001_verify_pair.json', 'execution/002_write_report.json'):
    record = read(relative)
    check('true_successful_execution_' + relative, record['returncode'] == 0 and Path(record['stdout_file']).is_file() and Path(record['stderr_file']).is_file())
result = dict(status='PASS' if all(c['status'] == 'PASS' for c in checks) else 'FAIL',
    generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), checks=checks,
    total_checks=len(checks), failed_checks=[c for c in checks if c['status'] == 'FAIL'],
    input_files=len(final), report_sha256=hashlib.sha256((OUT / 'EXPERIMENT_AUDIT.md').read_bytes()).hexdigest(),
    machine_report_sha256=hashlib.sha256((OUT / 'EXPERIMENT_AUDIT.json').read_bytes()).hexdigest(),
    note='Final serialization/trace/evidence/hash validation; no repeat neural or raw metric evaluation. Output manifest is generated after this child execution is logged.')
(OUT / 'final_artifact_validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('status', 'total_checks', 'failed_checks', 'input_files', 'report_sha256', 'machine_report_sha256')}, indent=2))
raise SystemExit(0 if result['status'] == 'PASS' else 1)
