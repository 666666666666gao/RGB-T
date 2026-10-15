"""Write the bounded reviewer verdict, evidence references, and full trace."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_inputs_1040')
OUT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_lasher_formal_audit_20261015')
TRACE = OUT / 'trace'
TRACE.mkdir(exist_ok=True)

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def evidence(rel, line, supports):
    return {'file': str(ROOT / rel).replace('\\', '/'), 'line': line, 'supports': supports}

verification = load(OUT / 'deterministic_verification.json')
static = load(OUT / 'static_verification.json')
report = load(ROOT / 'report/full_report.json')
before, after = load(OUT / 'input_hashes_before.json'), load(OUT / 'input_hashes_after.json')
assert verification['failed_comparisons'] == static['failed'] == 0 and before == after
now = datetime.now(timezone.utc).isoformat()

checks = {
    'A_ground_truth_provenance': {
        'status': 'PASS', 'classification': 'real_gt',
        'details': 'Within the supplied artifact boundary, the evaluator loads 245 dataset init.txt files independently of tracker predictions. All declared source-path hashes/sizes match. No noninitial prediction row exactly equals its GT row; no GT file equals a prediction file. First-frame GT initialization is explicit and expected. These observations reject a direct prediction-copy proxy in this bundle; they do not authenticate the original official distribution or prove the full absent runtime dependency graph.',
        'evidence': [evidence('project/cadtrack_abc/score.py', 36, 'LasHeR GT is loaded directly from dataset init.txt'), evidence('project/cadtrack_abc/infer.py', 56, 'Only the first annotation line initializes inference'), evidence('project/cadtrack_abc/infer.py', 65, 'Subsequent steps receive images/frame/state rather than GT'), evidence('raw_input_manifest.json', 4606, 'Declared actual dataset source root'), evidence('raw_input_manifest.json', 34, 'First dataset GT file provenance record')]
    },
    'B_score_normalization_and_protocol': {
        'status': 'PASS',
        'details': 'No score is normalized by model-output statistics. NPR normalizes each center coordinate using GT width/height plus 1e-8. The supplied installed toolkit is reproduced exactly, including float32 integer rounding, GT first-frame replacement, NPR-before-PR/SR, threshold inequalities, equal sequence weighting, and the invalid-GT sentinel behavior. Upstream official-version authenticity is unavailable, so this is a supplied-toolkit reproduction rather than an independently authenticated canonical protocol certification.',
        'evidence': [evidence('toolkit/rgbt/dataset/basedataset.py', 22, 'Float32 text loader output is rounded to integer coordinates'), evidence('toolkit/rgbt/utils.py', 118, 'NPR denominator is GT dimensions'), evidence('toolkit/rgbt/utils.py', 141, 'Normalized center error formula'), evidence('toolkit/rgbt/metrics/metrics.py', 323, 'NPR thresholds, first-frame and invalid-GT handling'), evidence('toolkit/rgbt/metrics/metrics.py', 280, 'LasHeR PR definition'), evidence('toolkit/rgbt/metrics/metrics.py', 236, 'LasHeR SR definition'), evidence('project/cadtrack_abc/score.py', 54, 'NPR, PR, SR execution order')]
    },
    'C_result_existence_numbers_and_lineage': {
        'status': 'WARN',
        'details': 'The completed LasHeR report, both shard receipts, all 245 prediction files and all 245 latency arrays exist and reconcile to 220703 frames / 220458 measured tracking calls. All standard scores, mean curves, per-sequence JSON/CSV scores, attribute scores and latency values match exactly. The timing concurrency description is incorrect for this relay. Selected epoch10 and checkpoint path/hash metadata agree across snapshots; the checkpoint bytes, original inference process receipts/logs and complete runtime provenance are not present for independent attestation.',
        'evidence': [evidence('report/full_report.json', 2, 'Completed LasHeR report, epoch and inventory'), evidence('report/full_report.json', 8, 'Three standard reported metrics'), evidence('report/full_report.json', 402, 'Incorrect two-GPU timing description'), evidence('predictions/complete_shard0.json', 748, '123 sequences / 107898 frames'), evidence('predictions/complete_shard1.json', 742, '122 sequences / 112805 frames'), evidence('selection_snapshot.json', 2, 'Selected epoch10 and DEV score'), evidence('main_observation_1040.json', 6, 'Selected checkpoint path and recorded SHA256'), evidence('raw_input_manifest.json', 4610, 'Images/checkpoint bytes were not copied')]
    },
    'D_called_vs_dead_metrics': {
        'status': 'PASS',
        'details': 'The scoring entry point calls NPR, PR and SR via getattr; LasHeR binds them to NPR, PR_LasHeR and SR_LasHeR. All outputs materialize in the report, CSV and reconstructed curves. summarize_lasher is called and has a result artifact. MPR/MSR, generic PR/SR and GTOT variants are other dataset branches and are not misrepresented as LasHeR results. Training cached_validation is called by training and explicitly labels its scores diagnostic. No claimed standard metric depends on uncalled code. Missing raw mechanism traces prevent full revalidation of mechanism outputs, not proof of dead code.',
        'evidence': [evidence('project/cadtrack_abc/score.py', 53, 'Registration and metric calls'), evidence('toolkit/rgbt/dataset/lasher_dataset.py', 19, 'Metric class bindings'), evidence('project/cadtrack_abc/score.py', 93, 'Mechanism summary is called and written'), evidence('project/cadtrack_abc/train_future_actions.py', 157, 'Cached validation call'), evidence('project/cadtrack_abc/train_future_actions.py', 60, 'Cached utility scores explicitly exclude formal metrics')]
    },
    'E_scope_and_claim_ceiling': {
        'status': 'WARN',
        'details': 'This bundle proves one selected search_actions epoch10 LasHeR evaluation across 245 sequences, with one recorded training seed42. It does not prove the full three-benchmark result, matched cost improvement, independently effective A/B/C mechanisms, or cross-seed robustness. One best seed is allowed by the current plan; extra successful seeds are not imposed here. PR and NPR are below the user-provided reference, SR is only +0.0419915 pp, and none of the +6 pp targets is met. Snapshots that still say formal_metrics_complete=false refer to prior stages or to completion of all three datasets; the completed LasHeR artifact is not thereby contradicted.',
        'evidence': [evidence('fit_complete.json', 61, 'Seed42'), evidence('documentation/plan_snapshot.json', 351, 'Current ACTIVE_UNMET acceptance scope'), evidence('documentation/plan_snapshot.json', 376, 'One best seed allowed; three successful seeds not required'), evidence('report/full_report.json', 1882, 'Reference deltas'), evidence('report/full_report.json', 1894, 'All three targets false'), evidence('main_observation_1040.json', 9, 'RGBT234 jobs still running at observation'), evidence('main_formal_relay.py', 116, 'Global formal completion only after all three datasets')]
    },
    'F_evaluation_type': {
        'status': 'PASS', 'classification': 'real_gt',
        'details': 'Standard tracking metrics use dataset annotations. Mechanism error/write diagnostics are explicitly GT-assisted localization proxies, not semantic identity ground truth or causal future evidence. They are not model-generated substitute GT. The full pipeline is not classified as synthetic_proxy or simulation_only.',
        'evidence': [evidence('project/cadtrack_abc/score.py', 36, 'Dataset GT scoring'), evidence('project/cadtrack_abc/mechanism_summary.py', 1, 'Post-hoc GT-assisted diagnostics'), evidence('report/mechanism_report.json', 7776, 'Different diagnostic valid-GT mask'), evidence('report/mechanism_report.json', 7778, 'Immediate harms are not a causal future comparison')]
    }
}

issues = [{
    'id': 'W-TIMING-01', 'severity': 'WARN', 'blocking': False,
    'title': 'Timing report incorrectly says concurrent two-GPU run',
    'finding': 'The report/score.py static string says concurrent two-GPU run. The supplied formal relay launches both inference shards with gpu=2; the handoff identifies both LasHeR processes on GPU2. A second GPU may run a different control task, but this is not two-GPU distribution of the audited tracker run.',
    'impact': 'Numeric latency/FPS arithmetic remains correct. The hardware concurrency label must not be used as evidence of two-GPU throughput or matched baseline speedup.',
    'recommended_correction': 'In a derived corrected report or future report metadata, describe two concurrent inference workers on physical GPU2, with shared-host concurrent workload; preserve this sealed original input. Keep FPS explicitly as reciprocal pooled worker-frame latency, not wall-clock throughput.',
    'evidence': [evidence('report/full_report.json', 402, 'Two-GPU phrase'), evidence('project/cadtrack_abc/score.py', 81, 'Hard-coded phrase'), evidence('main_formal_relay.py', 89, 'Two shard commands'), evidence('main_formal_relay.py', 92, 'Both commands routed to GPU2'), evidence('documentation/HANDOFF_snapshot.md', 5100, 'Both LasHeR NN process IDs stated on GPU2')]
}]

limitations = [
    {'id': 'L-UPSTREAM', 'status': 'UNAVAILABLE', 'check': 'Official-distribution authenticity and raw attribute-membership provenance', 'details': 'The supplied GT and toolkit source hashes establish integrity of collected files. No official release/archive checksum, upstream git revision, original lashertest.txt, or AttriSeqsTxt files are supplied. Membership is reproduced from toolkit_protocol.json exported by the provided exporter, so the 19 attribute scores are verified conditional on that export.', 'evidence': [evidence('export_toolkit_protocol.py', 8, 'Exports installed evaluator names/attributes'), evidence('toolkit/rgbt/dataset/lasher_dataset.py', 78, 'Original attribute text dependency'), evidence('toolkit_protocol.json', 251, 'Supplied exported membership')]},
    {'id': 'L-CHECKPOINT', 'status': 'PARTIAL', 'check': 'Selected-checkpoint provenance and actual execution', 'details': 'Epoch10 is the correct argmax of the six supplied candidate summary scores, and best/trained-best are the same. Metadata records SHA256 a14130fafbc7eb49a17a3834fe8a43855f3ed0833ba576e737e74d1579965ad8. Weight bytes, raw full98 development records/GT/predictions, training logs/state, and original LasHeR process/exit logs are absent. Strict reload, actual optimizer updates, actual loaded tensor identity and DEV scores are receipt-level evidence only; they were not independently rerun.', 'evidence': [evidence('selection_snapshot.json', 9, 'Six candidate summaries'), evidence('project/cadtrack_abc/select_future_actions.py', 42, 'Selection and copy rule'), evidence('main_observation_1040.json', 7, 'Recorded checkpoint digest'), evidence('raw_input_manifest.json', 4610, 'No checkpoint bytes copied')]},
    {'id': 'L-SOURCE', 'status': 'PARTIAL', 'check': 'Full runtime code provenance and no-later-GT causality', 'details': 'The actual inference entry point reads only first-frame GT, and supplied runtime functions accept causal state/image inputs. The 14-file relay seal omits active search_actions.py, ported_modules.py and mechanism_summary.py; counterfactual.py and vendor CADTrack sources are not in this bundle, and no actual launch-time source-digest map is supplied. Collection-time hashes must not be described as a complete immutable run-code attestation.', 'evidence': [evidence('main_formal_relay.py', 26, 'Limited source_names seal'), evidence('project/cadtrack_abc/search_actions.py', 10, 'Missing counterfactual dependency'), evidence('project/cadtrack_abc/search_actions.py', 99, 'Runtime calls advance_candidate'), evidence('project/cadtrack_abc/baseline.py', 7, 'Absent vendor source dependency'), evidence('project/cadtrack_abc/infer.py', 56, 'Visible first-frame-only annotation read')]},
    {'id': 'L-MECHANISM', 'status': 'PARTIAL', 'check': 'Per-frame candidate/commit claims', 'details': 'No *_mechanisms.json traces are supplied. Candidate recall/newly recalled acceptance, immediate keep-vs-chosen rescues/harms, alternative selections and geometry/template/memory commits cannot be reconstructed from final predictions alone. Their reported ratios and extra-forward receipt sums are internally consistent but not independently validated behavior. Final-output IoU, valid/failure counts and all 881 recovery events ARE recomputed exactly.', 'evidence': [evidence('project/cadtrack_abc/mechanism_summary.py', 54, 'Requires raw mechanism traces'), evidence('project/cadtrack_abc/mechanism_summary.py', 59, 'Keep boxes required for rescues/harms'), evidence('project/cadtrack_abc/mechanism_summary.py', 79, 'Candidates/commits needed for counters'), evidence('report/mechanism_report.json', 38, 'Recovery totals')]},
    {'id': 'L-COST', 'status': 'UNAVAILABLE', 'check': 'Matched baseline/system-cost or wall-clock speed claim', 'details': '245 arrays provide 220458 synchronized per-step intervals and all arithmetic is exact. The interval includes decoding, crops, network/decision/state updates; it excludes initialization, model loading, final output CPU extraction after the timer, diagnostics extraction and serialization. Sum of worker times is not elapsed concurrent-run time, and no matched baseline timing or raw baseline predictions are supplied.', 'evidence': [evidence('project/cadtrack_abc/infer.py', 60, 'Initialization precedes measured loop'), evidence('project/cadtrack_abc/infer.py', 63, 'Timing interval and synchronization'), evidence('project/cadtrack_abc/infer.py', 68, 'Output/diagnostics extraction after the timer'), evidence('report/full_report.json', 404, 'User reference only; no paired baseline')]},
    {'id': 'L-GOAL', 'status': 'NOT_ACHIEVED', 'check': 'Overall project objective', 'details': 'The completed LasHeR point scores fail all +6 pp targets. Other required benchmarks, effective three-module controls and matched system cost are not completed evidence in this bundle. The overall project objective is not certified; no assertion of failure of the entire future research program is made.', 'evidence': [evidence('report/full_report.json', 1894, 'All target flags false'), evidence('documentation/plan_snapshot.json', 387, 'R3 independent contributions unproved'), evidence('documentation/plan_snapshot.json', 389, 'Goal completion not proven')]}
]

protocol_detail = {
    'reference': 'Supplied installed /data/wangwj/RGBT_toolkit-master/src snapshot; upstream official release not authenticated',
    'ground_truth_dtype_and_load': 'float64 np.loadtxt(dataset/init.txt, delimiter=comma, ndmin=2), replacing constructor GT',
    'prediction_serialization': 'infer writes tab-delimited xywh at 3 decimal places',
    'prediction_loader': 'float32 load_text then NumPy round(0), then list of row arrays',
    'metric_order': ['NPR', 'PR', 'SR'],
    'initialization': 'Only first GT box initializes inference; every metric replaces loaded first prediction with GT[0] and retains that frame in the denominator',
    'invalid_prediction': 'Nonpositive width/height after rounding carries forward previous prediction; observed affected frames=0',
    'PR': 'Center Euclidean error <= integer thresholds 0..50; index20 = 20 pixels',
    'NPR': 'Center coordinates divided componentwise by GT width/height+1e-8; Euclidean error <= thresholds 0..0.5 step0.01; index20 = 0.20',
    'SR': 'Inclusive ltwh-to-ltrb IoU; strict IoU > thresholds 0..1 step0.05; arithmetic mean of all21 curve samples, not trapezoidal AUC or plain mean IoU',
    'invalid_ground_truth': 'Any one of x,y,w,h <=0 sets error/IoU to -1. This counts as precision/NPR success at every nonnegative threshold and SR failure; frame remains in denominator.',
    'invalid_gt_frames': 1260, 'invalid_gt_sequences': 52, 'nonpositive_width_or_height_frames': 533,
    'positive_dimensions_but_invalid_x_or_y_frames': 727,
    'precision_credit_from_invalid_frames_sequence_equal_pp': 0.5568053864558151,
    'credit_interpretation': 'Contribution to the reported PR/NPR numerators with original denominators, not a replacement score or bias estimate against a different missing-data protocol',
    'aggregation': 'Equal weight per sequence, regardless of frame count; same equal weighting within each exported attribute membership list',
    'protocol_changed_by_audit': False
}

claims = [
    {'id': 'C-STANDARD', 'claim': 'Completed epoch10 LasHeR PR/NPR/SR, all245 sequences', 'impact': 'supported_with_supplied_toolkit_provenance_limit'},
    {'id': 'C-ATTR', 'claim': '19 attribute scores and complete mean/per-sequence curves', 'impact': 'supported_conditional_on_exported_membership'},
    {'id': 'C-REFERENCE', 'claim': 'Deltas against user reference PR72.74/NPR69.16/SR57.84', 'impact': 'arithmetic_supported_no_protocol_equivalence_or_significance_attestation'},
    {'id': 'C-GOAL', 'claim': 'All three +6 pp targets achieved', 'impact': 'not_achieved_explicitly_reported_false'},
    {'id': 'C-TIMING', 'claim': 'Two-GPU run / matched speedup / complete-system efficiency', 'impact': 'two_GPU_label_incorrect_other_cost_claims_unsupported'},
    {'id': 'C-MECHANISM', 'claim': 'Independent causal efficacy of A/B/C', 'impact': 'unsupported_by_this_bundle'},
    {'id': 'C-THREE', 'claim': 'Same-checkpoint all-three-benchmark completion', 'impact': 'not_completed_in_supplied_observation'}
]

audit = {
    'audit_skill': 'experiment-audit', 'project': 'CADTrack ABC R1 selected search_actions epoch10 LasHeR',
    'date': '2026-10-15', 'generated_at': now, 'verdict': 'WARN', 'overall_verdict': 'warn', 'integrity_status': 'warn',
    'reason_code': 'recomputed_results_pass_with_timing_label_correction_and_provenance_limits',
    'summary': 'All completed LasHeR standard results exactly reproduce from supplied actual GT/predictions. One nonblocking timing-scope error and explicit evidence limits remain. The +6 pp objective is not achieved.',
    'blocking_issue_count': 0, 'blocking_issues': [], 'nonblocking_issue_count': len(issues),
    'blocking_scope': 'Integrity of the supplied completed LasHeR metric artifacts; does not mean the project objective or all stronger claims pass',
    'agent_id': '/root/r1_lasher_epoch10_result_integrity_audit', 'agent_id_kind': 'canonical_task_path',
    'verdict_id': 'r1_main_lasher_formal_audit_20261015',
    'executor_model': 'UNATTESTED', 'executor_family': 'openai',
    'requested_reviewer_model': 'gpt-6-astra', 'requested_reasoning_effort': 'max',
    'reviewer_model': 'UNATTESTED', 'reviewer_reasoning': 'UNATTESTED', 'actual_routing': 'UNATTESTED',
    'reviewer_family': 'openai', 'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'input_root': str(ROOT).replace('\\', '/'), 'checks': checks, 'issues': issues, 'limitations': limitations,
    'evaluation_type': 'real_gt', 'protocol': protocol_detail,
    'deterministic_verification': {'status': 'PASS', 'acceptance_scope': 'Arithmetic and byte-preservation checks only',
        'comparisons': verification['total_comparisons'] + static['comparisons'],
        'numeric_structural_and_hash_comparisons': verification['total_comparisons'],
        'additional_static_comparisons': static['comparisons'], 'failed_comparisons': 0,
        'per_sequence_curve_values_compared': 30135, 'overall_metric_values': 3,
        'mean_curve_values_compared': 123, 'per_sequence_metric_values_each_json_and_csv': 735,
        'attribute_metric_values_compared': 57, 'max_score_curve_latency_abs_error': 0.,
        'files': ['deterministic_verification.json', 'static_verification.json', 'verification_execution_receipt.json'],
        'method': 'CPU Python3.12.6/NumPy1.26.4; actual metric/loader bodies replayed in memory with imports separated, plus independent NumPy formula; original CLI and neural model not executed'},
    'overall_metrics_percent': verification['overall_metrics_percent'],
    'delta_percentage_points_vs_user_reference': report['delta_percentage_points_vs_user_reference'],
    'target_percent': report['lasher_stage_goal']['targets_percent'],
    'targets_reached': report['lasher_stage_goal']['per_metric_reached'],
    'overall_project_objective_certified': False,
    'timing_recomputed': verification['timing'], 'mechanism_independently_verified_subset': verification['verified_mechanism_subset'],
    'input_preservation': {'files': len(before), 'before_after_identical': before == after,
        'initial_manifest_identical': before == load(OUT / 'initial_input_hashes.json'),
        'source_manifest_records': 767, 'source_manifest_hash_size_mismatches': 0,
        'collection_source_manifest_missing_files': verification['source_manifest_does_not_cover']},
    'audited_input_hashes': {p: 'sha256:' + value for p, value in before.items()},
    'claims': claims, 'trace_path': str(TRACE).replace('\\', '/'),
    'privacy': 'Audit exports metrics, counts, hashes and evidence references only; no raw GT/prediction coordinates or latency arrays copied to audit outputs.'
}
dump(OUT / 'EXPERIMENT_AUDIT.json', audit)

md = '''# Experiment audit: R1 selected epoch10 LasHeR

**Overall verdict: WARN. Blocking issues: 0.** One concrete nonblocking reporting error requires correction. The completed standard LasHeR numbers pass deterministic reproduction; the overall project objective is **not achieved or certified**.

Requested reviewer: `gpt-6-astra`, reasoning `max`; actual model/effort routing: `UNATTESTED`. Fresh reviewer context, `review_independence: same-family`, `acceptance_status: provisional`. Date: 2026-10-15. This is a result audit of supplied files, not a new neural experiment.

All input references below are relative to `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_inputs_1040`. Derived evidence is in this audit directory. No input was modified.

## Verified result and claim ceiling

| Metric | Recomputed % | User reference % | Difference, pp | +6 pp target % | Reached |
|---|---:|---:|---:|---:|---|
| PR | 72.48674289849187 | 72.74 | -0.25325710150812597 | 78.74 | No |
| NPR | 68.84299897334117 | 69.16 | -0.31700102665882923 | 75.16 | No |
| SR | 57.88199150093345 | 57.84 | +0.04199150093344883 | 63.84 | No |

The three numbers, their deltas and target flags exactly match `report/full_report.json:8`, `:1882`, and `:1894`. The reference is explicitly user-provided with no raw paired baseline predictions or paired CI (`report/full_report.json:404`). Arithmetic comparison is supported; historical protocol equivalence, significance, speedup and independent module efficacy are not established.

The completed scope is **245 sequences / 220703 frames**, with **220458 post-initialization tracking calls**. Shard0 contains 123 sequences / 107898 frames and shard1 122 / 112805; the exact sorted alternating partitions, all sequence names, all shapes, finite data, all 245 first-box initializations and receipt frame/time sums agree (`predictions/complete_shard0.json:748`; `predictions/complete_shard1.json:742`; `project/cadtrack_abc/infer.py:48`). One training seed42 is recorded (`fit_complete.json:61`). The current plan permits one best seed and does not require three successful seeds (`documentation/plan_snapshot.json:376`).

## A. Ground-truth provenance — PASS within the supplied artifact boundary

The scorer reads actual dataset `init.txt` files (`project/cadtrack_abc/score.py:36`); it never uses baseline predictions as GT. The raw manifest identifies `/data/wangwj/dataset/LasHeR/testingset` as the source (`raw_input_manifest.json:4606`). All 245 source paths and every one of the 767 manifest file SHA256/size pairs match the collected files. GT arrays and prediction arrays are distinct; none of the 220458 noninitial prediction rows exactly equals its GT row, and no GT file byte hash equals a prediction file hash. This is corroboration against direct copying, not a universal proof against fabrication.

Inference reads only the first annotation line and then calls `model.step` with current images, frame index and state (`project/cadtrack_abc/infer.py:56`, `:65`). All serialized first boxes match the float32 GT initialization rounded to three decimals. The supplied source does not reveal later-GT-fed inference. Full no-leakage/runtime certification is unavailable because `counterfactual.py` and vendor code are absent (`project/cadtrack_abc/search_actions.py:10`, `:99`; `project/cadtrack_abc/baseline.py:7`). No neural code was executed in this review.

The original official download/archive hashes and original toolkit attribute text files are not included. Therefore the authenticity claim stops at the supplied dataset/source manifest; it is not an independent reauthentication of the official distribution.

## B. Score normalization and exact protocol — PASS

No metric denominator is the maximum, minimum or mean of model predictions. NPR uses **GT width/height + 1e-8** (`toolkit/rgbt/utils.py:118`, `:141`). The exact supplied installed-toolkit protocol is:

1. Inference serializes xywh predictions to three decimals (`project/cadtrack_abc/infer.py:86`). The toolkit loads them as float32 and applies `round(0)` (`toolkit/rgbt/utils.py:31`; `toolkit/rgbt/dataset/basedataset.py:22`). Raw GT is loaded as float64, bypassing constructor GT rounding (`project/cadtrack_abc/score.py:36`). Integer rounding changes 875755 coordinate entries.
2. Metrics execute **NPR → PR → SR** (`project/cadtrack_abc/score.py:54`). Each replaces prediction0 with GT0 and **retains the initial frame in its denominator**; the source comment “ignore the first frame” means replacement, not removal. Invalid predicted width/height would carry forward the preceding prediction; no supplied prediction triggers this rule.
3. PR uses center error `<= 20` pixels, with 51 thresholds 0…50. NPR uses normalized center error `<= 0.20`, with 51 thresholds 0…0.5 in steps of0.01. The representative score is threshold index20 (`toolkit/rgbt/metrics/metrics.py:280`, `:323`).
4. SR uses the inclusive-coordinate IoU implementation and strict `IoU > threshold`, at 21 thresholds 0…1. Its reported score is the **arithmetic mean of all21 samples**, not trapezoidal AUC and not plain mean IoU (`toolkit/rgbt/utils.py:162`; `toolkit/rgbt/metrics/metrics.py:236`).
5. Each sequence has equal weight regardless of length. Attribute scores use the same equal weighting over exported member sequences (`project/cadtrack_abc/score.py:60`, `:65`).
6. **Native invalid-GT behavior is material:** any x/y/w/h `<=0` sets center error / IoU to `-1`. That counts as success for PR/NPR at every nonnegative threshold and failure for SR; such frames are not dropped (`toolkit/rgbt/metrics/metrics.py:266`, `:309`, `:351`). There are1260 affected frames in52 sequences:533 with nonpositive dimensions and727 with positive dimensions but nonpositive x/y. Their contribution to sequence-equal PR/NPR numerators is0.5568053864558151 pp with original denominators. This is a description of the preserved protocol, not a new score or a correction to published benchmark conventions.

The audit preserved every rule. It does not certify that this collected toolkit snapshot is an untouched upstream official release. All19 attribute scores match **conditional on `toolkit_protocol.json:251` membership**. The exporter reads installed evaluator membership (`export_toolkit_protocol.py:8`); original `AttriSeqsTxt` files used at `toolkit/rgbt/dataset/lasher_dataset.py:78` are absent.

## C. Result existence, arithmetic and execution provenance — WARN

Both completion receipts,245 prediction files,245 latency arrays, `full_report.json`, `per_sequence.csv`, and `mechanism_report.json` exist. Replaying the supplied metric and loader bodies, then separately implementing the equations in NumPy, yields exact equality for **30135 per-sequence curve values,123 mean-curve values,735 per-sequence scores in each of JSON and CSV,57 attribute metrics and3 overall metrics**. All threshold arrays match. There are41963 numeric/structural/hash comparisons plus9 static checks: **41972 comparisons,0 failures**, maximum score/curve/latency discrepancy0.0. See `deterministic_verification.json`, `static_verification.json`, and `verification_execution_receipt.json`.

The audit replayed original mathematical class/function bodies after separating imports, avoiding absent dataset/plotting dependencies. It did not execute the original full CLI, load checkpoints, use GPU/SSH, run neural inference, or install packages. An independent NumPy implementation agrees with every curve value. Python3.12.6 and preexisting NumPy1.26.4 were used. The initial import attempt lacked NumPy on its path; the genuine `ModuleNotFoundError` is retained in `audit_tool_attempts.json`. Only the review process PYTHONPATH was set to the existing cache; no environment files were changed.

Epoch10 is the actual maximum of the six candidate summaries, and also the best positive epoch (`selection_snapshot.json:2`, `:9`, `:47`; selection rule at `project/cadtrack_abc/select_future_actions.py:42`). The best path and recorded SHA `a14130fafbc7eb49a17a3834fe8a43855f3ed0833ba576e737e74d1579965ad8` agree with the observation and plan (`main_observation_1040.json:6`; `documentation/plan_snapshot.json:853`). The fit receipt's1600 updates equals20×ceil(5092/64). These checks establish metadata consistency, not independent reproduction of training, DEV IoUs, optimizer state, actual loaded weights or strict reload. The weight bytes, raw DEV records and original LasHeR native process/exit logs are absent; `raw_input_manifest.json:4610` explicitly says no images/checkpoints were copied.

The relay's14-file source seal is narrower than the active dependency graph: it omits `search_actions.py`, `ported_modules.py` and `mechanism_summary.py`; the collected bundle also lacks `counterfactual.py` and CADTrack vendor sources. Collection-time hashes must not be promoted to complete launch-time/runtime provenance (`main_formal_relay.py:26`; `static_verification.json`). No active source was changed to address this audit limitation.

### W-TIMING-01: incorrect hardware-scope text

`report/full_report.json:402` and `project/cadtrack_abc/score.py:81` say “Concurrent two-GPU run”. This relay constructs both shard commands and runs them with `gpu=2` (`main_formal_relay.py:89`, `:92`); the handoff identifies both LasHeR processes on GPU2 (`documentation/HANDOFF_snapshot.md:5100`). Correct the derived presentation to **two concurrent inference workers on physical GPU2**, acknowledging shared-host concurrent work. Preserve the original sealed report.

All245 latency arrays and every per-sequence receipt sum reproduce exactly:

| Quantity | Recomputed |
|---|---:|
| Measured post-initialization calls | 220458 |
| Sum of worker-frame seconds | 23766.367712457053 |
| Mean latency, ms | 107.80451474864624 |
| P50 latency, ms | 106.1192315100925 |
| P95 latency, ms | 172.8491481437231 |
| Reciprocal mean latency, FPS | 9.27604935963554 |

FPS is calls divided by summed worker-frame time, **not elapsed concurrent-run throughput**. The timed region includes decoding/crops/forwards/decisions/state updates and synchronizes CUDA; initialization, model loading, final box CPU extraction after the timer, diagnostic extraction and file serialization are excluded (`project/cadtrack_abc/infer.py:60`, `:63`, `:68`, `:86`). This does not prove matched baseline or complete-system cost improvement.

## D. Called versus dead metrics — PASS

`score.py:53` registers the tracker; `:54`–`:57` calls NPR/PR/SR, and `lasher_dataset.py:19` binds the exact LasHeR metric classes. Their standard result keys and all computed curves exist. `score.py:93` calls the mechanism summarizer and writes its output. Generic PR/SR, MPR/MSR and GTOT variants serve other dataset paths; their absence from this LasHeR report is correct, not a phantom result. `train_future_actions.py:157` calls cached validation, whose `:60` scope explicitly denies formal-metric status.

The mechanism summary can be **partially** independently checked. From raw final predictions/GT, this review exactly reconstructs219925 valid post-initialization GT frames,70077 failure frames, sequence-equal IoU0.5894167200173702, frame-equal IoU0.5120501208799806, all881 failure/recovery events,792 recoveries,89 right-censored events and median recovery delay11 frames. The diagnostic mask intentionally differs from official scores (`report/mechanism_report.json:7776`).

No raw `*_mechanisms.json` files are supplied. Candidate recall, newly recalled acceptance, same-state rescues/harms, alternative selections and commit/pollution counters require those records (`project/cadtrack_abc/mechanism_summary.py:54`, `:59`, `:79`). Their ratios and total7380 extra forwards agree internally with receipt summaries, but that is not independent verification of per-frame behavior or actual visual execution. These unavailable checks are recorded as PARTIAL, never PASS by implication.

## E. Scope — WARN

The result supports one selected fixed-model LasHeR point estimate. PR/NPR are below the user reference, SR rises only0.0419915 pp; none of the +6 pp targets is met. The observed controller is still running RGBT234 and has only a completed LasHeR report (`main_observation_1040.json:5`, `:9`, `:60`). This bundle supplies no completed RGBT210/RGBT234 result for this checkpoint, matched-cost experiment or effective A/B/C control result. Overall completion remains unsupported (`documentation/plan_snapshot.json:351`, `:387`, `:389`).

`fit_complete.json:53` and `selection_snapshot.json:55` retain earlier/full-pipeline completion flags. They do not negate the newer completed LasHeR report: the relay sets global formal completion only after all three datasets (`main_formal_relay.py:116`). The handoff snapshot ends before the10:40 result observation. Historical +2/LasHeR-only plans are superseded by the explicit current acceptance section, not silently rewritten.

## F. Evaluation type — real_gt / PASS

Standard PR/NPR/SR use dataset annotations and are `real_gt`. Post-hoc mechanism diagnostics use GT-assisted localization proxies and explicitly deny semantic identity or causal future-effect interpretation (`report/mechanism_report.json:7776`, `:7778`; `project/cadtrack_abc/mechanism_summary.py:118`). They are not prediction-generated replacement GT.

## Preservation and actions

All778 enumerated input SHA256 values agree with the initial manifest and remain unchanged before/after verification. All767 collected source-manifest hashes and byte counts match. The11 files outside that source manifest are separately sealed by the initial/before/after hash manifests; this includes snapshots, exporter and audit-manifest metadata, not missing raw GT/predictions. No raw GT/prediction coordinates or latency arrays are emitted in these audit outputs.

1. Correct the timing concurrency description in derived reporting; keep original evidence immutable.
2. Report the exact supplied-toolkit semantics and the limited provenance of exported attribute membership.
3. Limit validated mechanism claims to the reconstructed subset; obtain the existing raw traces before auditing other mechanism counters.
4. Preserve the selected weight/source/DEV/native receipts for any stronger lineage claim. No new neural run is requested by this audit.
5. Keep the overall objective unmet until corresponding completed benchmark, target and mechanism/cost evidence exists.

Artifacts: `EXPERIMENT_AUDIT.json`, `deterministic_verification.json`, `static_verification.json`, `verified_per_sequence.csv`, `verified_per_sequence_curves.csv`, `verified_attributes.csv`, `input_hashes_before.json`, `input_hashes_after.json`, `source_manifest_verification.json`, `verification_execution_receipt.json`, and `trace/`. Deterministic arithmetic acceptance does not upgrade this same-family semantic review from provisional status.
'''
for old, new in {
    'all21': 'all 21', 'All245': 'All 245', 'All19': 'All 19',
    'all245': 'all 245', 'All778': 'All 778', 'All767': 'All 767',
    'step0.01': 'step 0.01', 'steps of0.01': 'steps of 0.01',
    'are1260': 'are 1260', 'in52': 'in 52', 'sequences:533': 'sequences: 533',
    'and727': 'and 727', 'is0.556': 'is 0.556', 'are41963': 'are 41963',
    'plus9': 'plus 9', 'comparisons,0': 'comparisons, 0', 'values,123': 'values, 123',
    'values,735': 'values, 735', 'CSV,57': 'CSV, 57', 'and3': 'and 3',
    'discrepancy0.0': 'discrepancy 0.0', 'receipt\'s1600': 'receipt\'s 1600',
    'equals20': 'equals 20', 'relay\'s14': 'relay\'s 14',
    'reconstructs219925': 'reconstructs 219925', 'frames,70077': 'frames, 70077',
    'IoU0.589': 'IoU 0.589', 'IoU0.512': 'IoU 0.512', 'all881': 'all 881',
    'events,792': 'events, 792', 'recoveries,89': 'recoveries, 89',
    'delay11': 'delay 11', 'total7380': 'total 7380', 'only0.041': 'only 0.041',
    'the10:40': 'the 10:40', 'The11': 'The 11', 'seed42': 'seed 42',
    'epoch10': 'epoch 10', 'Epoch10': 'Epoch 10', 'prediction0': 'prediction 0',
    'GT0': 'GT 0', 'index20': 'index 20', 'Python3.12.6': 'Python 3.12.6',
    'NumPy1.26.4': 'NumPy 1.26.4', 'Shard0': 'Shard 0', 'shard1': 'shard 1',
    'GPU2': 'GPU 2'
}.items():
    md = md.replace(old, new)
(OUT / 'EXPERIMENT_AUDIT.md').write_text(md, encoding='utf-8')

request = '''Use experiment-audit as a fresh experiment integrity reviewer. Read C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md. Be adversarial and draw conclusions from primary files only. Do not use SSH, GPU, model inference, install packages, modify any supplied input or running code, send external messages, or create agents. CPU deterministic calculations with existing E:/python.exe -X utf8 -B and NumPy are allowed. All input files are enumerated in C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_paths.json; inspect the paths and read the files directly, including evaluation code/toolkit source, raw actual GT, raw predictions/latencies, report, per-sequence CSV, protocol attribute membership/thresholds, execution receipts, snapshots of configuration/selection/fit/claims, and source provenance hashes. Input root: C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_inputs_1040. Output only into C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_lasher_formal_audit_20261015. Audit A-F: actual dataset GT provenance vs prediction-derived proxies; metric normalization and precise official protocol; actual execution/result existence and numbers; called vs dead metrics; scope; evaluation type. Independently recompute all completed standard overall and per-sequence scores/curves and attribute metrics from supplied GT/predictions according to the actual toolkit, where possible. Check latency arithmetic, receipt scope, cohort/frame counts, initialization, rounding, invalid-GT handling, selected-checkpoint provenance and score/claim consistency. Classify any unsupported or unavailable checks honestly. Do not alter official protocol to improve scores. Do not publish raw GT, predictions, coordinates, or raw latency arrays; keep them private. Produce EXPERIMENT_AUDIT.json and EXPERIMENT_AUDIT.md, deterministic verification details/CSV as useful, input before/after hashes, and request/response trace. The initial input hashes are already in output/initial_input_hashes.json. Use requested model gpt-6-astra/reasoning max, actual routing UNATTESTED unless tool evidence establishes it, review_independence same-family, acceptance_status provisional. Report exact file:line evidence, verdict PASS/WARN/FAIL and blocking_issue_count. Do not certify the overall project objective without corresponding completed evidence. Send concise completion findings and artifact paths.'''

dump(TRACE / 'run.meta.json', {'skill': 'experiment-audit', 'run_id': 'r1_main_lasher_formal_audit_20261015',
    'started_at': None, 'start_time_attestation': 'Unavailable in delegated context; no timestamp fabricated',
    'completed_at': now, 'executor': 'codex', 'executor_model': 'UNATTESTED', 'executor_family': 'openai',
    'requested_model': 'gpt-6-astra', 'requested_reasoning_effort': 'max', 'actual_routing': 'UNATTESTED',
    'review_independence': 'same-family', 'acceptance_status': 'provisional', 'project_dir': str(ROOT)})
dump(TRACE / '001-result-integrity.request.json', {'call_number': 1, 'purpose': 'round-1-result-integrity',
    'timestamp': None, 'tool': 'spawn_agent', 'model': 'gpt-6-astra', 'reasoning_effort': 'max',
    'actual_routing': 'UNATTESTED', 'files_referenced': list(before), 'prompt': request,
    'trace_note': 'Exact received delegated task; full file paths expanded from its path-only manifest. Native spawn return metadata is not visible in this reviewer context.'})
(TRACE / '001-result-integrity.response.md').write_text(md, encoding='utf-8')
dump(TRACE / '001-result-integrity.meta.json', {'call_number': 1, 'purpose': 'round-1-result-integrity',
    'timestamp': now, 'agent_id': audit['agent_id'], 'agent_id_kind': 'canonical_task_path',
    'model': 'UNATTESTED', 'reasoning_effort': 'UNATTESTED', 'requested_model': 'gpt-6-astra',
    'requested_reasoning_effort': 'max', 'reviewer_family': 'openai', 'review_independence': 'same-family',
    'acceptance_status': 'provisional', 'status': 'ok', 'verdict': 'WARN', 'blocking_issue_count': 0,
    'response_type': 'Full substantive reviewer report, identical to EXPERIMENT_AUDIT.md'})
(TRACE / 'events.jsonl').write_text(json.dumps({'event': 'review_trace', 'skill': 'experiment-audit',
    'purpose': 'round-1-result-integrity', 'agent_id': audit['agent_id'], 'trace_path': str(TRACE), 'status': 'ok'}) + '\n', encoding='utf-8')
dump(OUT / 'audit_tool_attempts.json', [
    {'attempt': 1, 'tool': 'exec_command', 'chunk_id': 'afe987', 'exit_code': 1,
     'action': 'E:/python.exe -X utf8 -B inspect_inputs.py without additional PYTHONPATH',
     'stderr': "Traceback (most recent call last):\n  File \"C:\\Users\\gb\\.codex_tmp\\cadtrack_abc_20261013\\r1_main_lasher_formal_audit_20261015\\inspect_inputs.py\", line 8, in <module>\n    import numpy as np\nModuleNotFoundError: No module named 'numpy'\n",
     'scope': 'Reviewer environment-resolution failure; not an experiment failure; no package installation'},
    {'attempt': 2, 'tool': 'exec_command', 'chunk_id': '292b0a', 'exit_code': 0,
     'action': 'Same interpreter/script with process-only PYTHONPATH pointing to existing NumPy1.26.4 cache',
     'numpy_path': 'D:/Program Files/UserCache/gb/uv/archive-v0/4TmToZ9eHxHD81sohMJS9',
     'result': 'Inventory and before-hash verification passed'},
    {'attempt': 3, 'tool': 'exec_command/write_stdin', 'start_chunk_id': '5dd8cf', 'completion_chunk_id': '3eb474',
     'native_session_id': 63640, 'exit_code': 0, 'action': 'verify_results.py', 'result': '41963 comparisons,0 failures',
     'stdout_file': 'verification.stdout.txt', 'stderr_file': 'verification.stderr.txt', 'receipt_file': 'verification_execution_receipt.json'},
    {'attempt': 4, 'tool': 'exec_command', 'chunk_id': '38ea6e', 'exit_code': 0,
     'action': 'verify_static.py', 'result': '9 static comparisons,0 failures'}
])
manifest = {str(path.relative_to(OUT)).replace('\\', '/'): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
            for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'audit_artifact_hashes.json'}
dump(OUT / 'audit_artifact_hashes.json', manifest)
print(json.dumps({'verdict': audit['verdict'], 'blocking_issue_count': 0,
    'deterministic_comparisons': audit['deterministic_verification']['comparisons'],
    'report_json': str(OUT / 'EXPERIMENT_AUDIT.json'), 'report_markdown': str(OUT / 'EXPERIMENT_AUDIT.md'),
    'report_json_sha256': hashlib.sha256((OUT / 'EXPERIMENT_AUDIT.json').read_bytes()).hexdigest(),
    'report_markdown_sha256': hashlib.sha256((OUT / 'EXPERIMENT_AUDIT.md').read_bytes()).hexdigest()}, indent=2))
