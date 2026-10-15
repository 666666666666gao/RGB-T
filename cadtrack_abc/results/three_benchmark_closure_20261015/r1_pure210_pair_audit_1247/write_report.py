"""Write the fresh reviewer's qualified findings around the independent results."""
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
LOGICAL_OUT = 'C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_pure210_pair_formal_audit_20261015'
INPUT = 'C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230'
AGENT = '/root/r1_pure210_pair_result_integrity_audit'
def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))
def write(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
def evidence(path, line, detail):
    return dict(path=INPUT + '/' + path, line=line, detail=detail)

results = read('recomputed_results.json')
verification = read('deterministic_verification.json')
bootstrap = read('paired_bootstrap_5000.json')
timing = read('aggregate_latency_statistics.json')
identity = read('checkpoint_identity_verification.json')
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
checks = {
 'A_gt_provenance': dict(status='PASS', qualification='Supplied-byte/data-flow provenance; publisher authentication is UNATTESTED.',
  details='Both roles are scored against the same 210 supplied dataset init.txt annotations. The scoring loader replaces the toolkit default GT with the dataset root. Main inference reads only the first annotation; pure inference reads labels for length checking and supplies only labels[0] to initialization. Later frame tracking takes images and prior state, not later GT. All 104706 rows are read and preserved; three rows with nonpositive width/height are kept under the native protocol.',
  evidence=[evidence('project/cadtrack_abc/score.py', 36, 'Dataset annotations replace evaluator.seqs_gt; RGBT210 uses init.txt for both streams.'),
   evidence('toolkit/rgbt/dataset/basedataset.py', 9, 'GT files are loaded independently of prediction files.'),
   evidence('project/cadtrack_abc/infer.py', 56, 'First annotation initializes; step at line65 receives the current image pair and state.'),
   evidence('project/cadtrack_abc/infer_baseline.py', 35, 'All labels are read for frame-count agreement, but only labels[0] is passed to initialize.'),
   evidence('project/cadtrack_abc/search_actions.py', 63, 'Runtime search and advance_candidate use causal state and current images.'),
   evidence('documentation/plan_snapshot.json', 1379, 'Collection declaration identifies the archive and GT inventory; archive bytes/publisher attestation are absent here.')]),
 'B_score_normalization': dict(status='PASS',
  details='No denominator is derived from model-output maxima, minima, means or confidence scores. GT/predictions are loaded as float32; predictions are rounded to integer coordinates, with ties-to-even, then the first predicted row is replaced by first GT. Strict center locations use (w-1)/2+x and integer-inclusive IoU uses x+w-1. PR uses <= at thresholds0..50 and reports20; SR uses > at21 linspace thresholds0..1 and reports the arithmetic mean, not a trapezoidal integral. Every original frame remains in the per-sequence denominator, including initialization and invalid annotations. Sequences are equally weighted. Independent vector arithmetic and the isolated supplied scalar PR/SR source produce identical full per-sequence curves.',
  evidence=[evidence('toolkit/rgbt/utils.py', 31, 'Default annotation/result dtype is np.float32.'),
   evidence('toolkit/rgbt/dataset/basedataset.py', 22, 'Predictions round(0) before ltwh conversion.'),
   evidence('toolkit/rgbt/utils.py', 122, 'CLE strict pixel-center definition.'),
   evidence('toolkit/rgbt/utils.py', 162, 'Integer-inclusive overlap and ordinary union denominator.'),
   evidence('toolkit/rgbt/metrics/metrics.py', 177, 'PR initialization replacement, <=, full len(res), sequence mean.'),
   evidence('toolkit/rgbt/metrics/metrics.py', 212, 'SR initialization replacement, >, full len(res), mean over sequences/thresholds.'),
   evidence('project/cadtrack_abc/score.py', 54, 'RGBT210 invokes only PR/SR, scales fractions by100, derives sequence and attribute metrics.')]),
 'C_result_existence': dict(status='WARN',
  details='All supplied raw files, 420 complete prediction series, 420 latency arrays, four shard receipts, both JSON/CSV reports, mean curves and12 attribute aggregates reproduce exactly. Main best.pth SHA and epoch10 literal match state/selection/report; the pure checkpoint has collection-time SHA metadata but no bytes. Recorded child exits and completion states agree. Stale baseline_source/timing prose is present in both raw reports. Historical imports, loaded parameter tensors and original controller OS exit are not independently attested by this static bundle. Development scores and20-epoch training are receipt-consistent, not independently reconstructed from missing training/development raw artifacts.',
  evidence=[evidence('results/main/rgbt210/full_report.json', 2, 'Completed main epoch10 report and original denominators.'),
   evidence('results/pure_rgbt210/full_report.json', 2, 'Completed pure epoch14 report and original denominators.'),
   evidence('main_state.json', 8, 'Recorded main checkpoint and configuration hashes checked against supplied bytes.'),
   evidence('main_selection.json', 2, 'Selected epoch10 is the maximum of six declared DEV candidate means; best/trained_best coincide.'),
   evidence('main_fit_complete.json', 7, '20 epochs/1600 updates and5092 queries; training seed42 is recorded at line61.'),
   evidence('pure_checkpoint_identity.json', 2, 'External checkpoint path/size/hash; explicitly collection-time-only, bytes not collected.'),
   evidence('pure210_state.json', 51, 'Both inference child exits0 and score exit0.'),
   evidence('documentation/run_pure_rgbt210_gpu2_remote.stdout.txt', 2, 'Final original stdout state agrees exactly with pure210_state.json; stderr empty.'),
   evidence('results/pure_rgbt210/full_report.json', 239, 'Stale two-GPU timing wording.'),
   evidence('results/pure_rgbt210/full_report.json', 241, 'Stale no-baseline-predictions/no-paired-CI wording.')]),
 'D_dead_code': dict(status='PASS',
  details='The claimed PR/SR functions are on the actual supplied scoring call path, have corresponding complete output fields, and were independently executed offline as isolated numeric definitions. MPR/MSR, GTOT and LasHeR helpers in metrics.py apply to other protocols and are not used for this RGBT210 claim. Training data.iou, future-action utility and synthetic sanity helpers are not substituted for the reported PR/SR. This confirms call-path consistency, not historical import attestation.',
  evidence=[evidence('project/cadtrack_abc/score.py', 53, 'Registers raw prediction directory, invokes dataset-specific metrics, writes results at line97.'),
   evidence('toolkit/rgbt/dataset/rgbt210_dataset.py', 86, 'PR dispatches to PR_fun.'),
   evidence('toolkit/rgbt/dataset/rgbt210_dataset.py', 113, 'SR dispatches to SR_fun.'),
   evidence('documentation/run_pure_rgbt210_gpu2_remote.py', 57, 'Controller invokes cadtrack_abc.score and records successful completion.'),
   evidence('main_state.json', 417, 'Recorded RGBT210 scoring command and return code.'),
   evidence('project/cadtrack_abc/future_pipeline.py', 263, 'Generic pipeline inference/scoring call chain; this source is not proof that the generic owner was the particular historical controller.')]),
 'E_scope': dict(status='WARN',
  details='Evidence establishes the fixed selected main checkpoint and the recorded pure baseline on one common RGBT210 benchmark, with210 equally weighted sequences and104706 frame pairs per role. Main has lower point estimates and both marginal paired5000 percentile intervals include zero. Neither superiority, significant inferiority, equivalence, three independently effective modules, matched speedup, training-seed stability nor research-goal completion follows. Main seed42 is recorded; the pure pretrained training seed is not supplied. Repeated98-DEV selection is conditioned on rather than included in uncertainty. No external control-job status is inferred from the static bundle.',
  evidence=[evidence('main_fit_complete.json', 58, 'Recorded main training hyperparameters, seed42.'),
   evidence('main_selection.json', 56, 'Repeated98 development selection explicitly acknowledged.'),
   evidence('documentation/HANDOFF_snapshot.md', 5, 'Current overview says the research targets are unmet.'),
   evidence('documentation/HANDOFF_snapshot.md', 7, 'Current overview already rejects matched baseline speed and independent module-effect claims.'),
   evidence('documentation/plan_snapshot.json', 385, 'R1/R2/R3 and independent-contribution requirements are separate stages.'),
   evidence('main_state.json', 316, 'Historical RGBT210 inference uses two workers on physical GPU2.'),
   evidence('pure210_state.json', 7, 'Pure run likewise records physical GPU2 and two workers, in a different time window.')]),
 'F_evaluation_type': dict(status='PASS', classification={'main': 'real_gt', 'pure': 'real_gt'},
  details='Both evaluations compare predictions with supplied dataset-provided annotation files. These are real_gt evaluations within the supplied-byte provenance limit, not synthetic agreement/proxy tests. Initial-frame GT use is the benchmark initialization convention, and subsequent labels are used for offline scoring.',
  evidence=[evidence('project/cadtrack_abc/score.py', 38, 'RGBT210 ground-truth annotation paths.'),
   evidence('toolkit/rgbt/dataset/basedataset.py', 12, 'Dataset annotation file reads.'),
   evidence('raw_input_manifest.json', 4, 'Supplied annotation source paths, lengths and hashes; all1147 manifest entries verified.')])}

issues = [
 dict(id='N1', severity='nonblocking', category='stale_report_metadata',
      detail='Both reports line239 say concurrent two-GPU run, but both controller records declare two inference workers on GPU2. Both line241 say no baseline predictions rerun and no paired CI; this is stale template prose for the newly completed pure210 role and does not describe this new paired audit.',
      action='Keep raw reports immutable. In derived summaries identify the roles, recorded two-workers-on-GPU2 execution, timing scope, and this paired-bootstrap artifact explicitly.'),
 dict(id='N2', severity='nonblocking', category='historical_runtime_attestation',
      detail='Main checkpoint bytes and source/config snapshots are verifiable, but no independent historical loaded-state/import trace is supplied. Pure checkpoint bytes are absent and its SHA is a collection-time receipt. Strict-load flags and250 loaded tensors are declarations supported by source, not a new neural replay.',
      action='Qualify historical runtime identity as UNATTESTED; retain the supplied hash/receipt evidence without claiming more.'),
 dict(id='N3', severity='nonblocking', category='publisher_authentication',
      detail='The dataset collection manifest and210 GT files agree with the seal. Original archive bytes, image files, independent publisher hashes/signatures and the installed toolkit default GT distribution are not in the bundle. Archive URL/hash and historical agreement with installed GT are recorded declarations.',
      action='Say supplied RGBT210 dataset annotations and supplied toolkit protocol; do not imply independent authentication of the original publisher distribution.'),
 dict(id='N4', severity='nonblocking', category='scope_and_selection',
      detail='One fixed main checkpoint and one pretrained pure role do not establish seed stability or any independent module contribution. Full98 DEV histories, raw training logs, main LasHeR/RGBT234 raw evaluation bundles, and external control-job evidence are outside this pair audit.',
      action='Restrict numerical claims to this fixed RGBT210 pair; keep the research goal unmet and external control status unassessed.'),
 dict(id='N5', severity='nonblocking', category='bootstrap_dependence',
      detail='Sequence bootstrap conditions on selected checkpoints and treats named sequences as exchangeable. Frames within videos and potentially related events across videos are dependent. Marginal95% percentile intervals do not represent training-seed variation or simultaneous95% joint coverage.',
      action='Publish the order/seed/replicates and dependence/selection qualifications with every interval.'),
 dict(id='N6', severity='nonblocking', category='review_attribution',
      detail='Requested gpt-6-astra/max/fork-none is recorded; this reviewer has no independent host attestation of actual model, reasoning effort or fork routing.',
      action='Keep actual model/routing/effort UNATTESTED and review_independence same-family, acceptance_status provisional.')]

claims = [
 dict(id='C1', claim='The two completed supplied RGBT210 reports are numerically correct.', impact='supported', qualification='Under the exact supplied protocol, with the original210 sequences and104706 frame denominators.'),
 dict(id='C2', claim='Main-minus-pure fixed-checkpoint PR/SR differences and paired sequence intervals.', impact='supported_with_qualifiers', qualification='Paired5000 seed20261015, toolkit sequence order, PCG64 and linear marginal percentiles; see paired_bootstrap_5000.json.'),
 dict(id='C3', claim='Main improves RGBT210 accuracy over pure CAD.', impact='unsupported', qualification='Both point differences are negative; both95% intervals include zero.'),
 dict(id='C4', claim='The main and pure models are equivalent, or main is significantly inferior.', impact='unsupported', qualification='Intervals crossing zero establish neither equivalence nor a significant difference.'),
 dict(id='C5', claim='All three modules independently contribute gains.', impact='unsupported', qualification='No full factorial or matched individual-module controls in this pair.'),
 dict(id='C6', claim='Main provides matched speedup or lower system cost.', impact='unsupported', qualification='Recorded worker latency aggregates are not a matched benchmark or wall-clock concurrent throughput.'),
 dict(id='C7', claim='Training-seed stability or robust repeated-training gains.', impact='unsupported', qualification='Sequence resampling does not resample training runs.'),
 dict(id='C8', claim='Original publisher data or exact historically loaded parameters/imports authenticated.', impact='unattested'),
 dict(id='C9', claim='Current external control-job state.', impact='not_assessed', qualification='Static snapshots are not live status.'),
 dict(id='C10', claim='The overall research goal is achieved.', impact='unsupported'),
 dict(id='C11', claim='Main best.pth bytes match the recorded selected epoch10 identity.', impact='supported_with_qualifiers', qualification='SHA256 and unique pickle epoch literal verified without torch import/unpickling; not historical load attestation.')]

audit = dict(audit_skill='experiment-audit', verdict='WARN', overall_verdict='warn', integrity_status='warn',
 reason_code='verified_fixed_pair_with_provenance_scope_and_metadata_limits',
 summary='All4154 deterministic checks pass; both210-sequence reports reproduce exactly. Main-minus-pure point differences are negative and both paired95% sequence intervals include zero. Metadata/provenance/scope limits remain; zero blocking findings for qualified artifact-level accuracy reporting.',
 date='2026-10-15', generated_at=now, project='CADTrack ABC R1 main / pure CAD RGBT210 pair',
 agent_id=AGENT, verdict_id='r1_pure210_pair_20261015',
 requested_reviewer_model='gpt-6-astra', requested_reviewer_reasoning='max', requested_fork_turns='none',
 reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', actual_routing='UNATTESTED', actual_fork_mode='UNATTESTED',
 executor_model='UNATTESTED', executor_family='openai', reviewer_family='openai',
 review_independence='same-family', acceptance_status='provisional',
 deterministic_evidence_status='accepted_for_supplied_bytes_and_protocol_only',
 input_root=INPUT, output_root=LOGICAL_OUT, trace_path=LOGICAL_OUT+'/.aris/traces/experiment-audit/2026-10-15_run01',
 runtime=verification['runtime'], scope=dict(dataset='rgbt210', sequences_per_role=210, frames_per_role=104706,
     latency_samples_per_role=104496, main_epoch=10, pure_epoch_declared=14, main_training_seed_declared=42,
     pure_training_seed='UNATTESTED', neural_execution=False, network=False, ssh=False, gpu_queries=False, package_installs=False),
 deterministic_verification=dict(status=verification['status'], named_checks=verification['total_checks'], failures=len(verification['failed_checks']),
     input_files=verification['input_files'], collection_manifest_entries=verification['collection_manifest_entries'], source_python_files_parsed=verification['source_python_files'],
     report_metric_and_curve_max_absolute_error=0.0, path=LOGICAL_OUT+'/deterministic_verification.json'),
 audited_input_hashes=read('input_hashes_before.json'),
 checks=checks, roles=results['roles'], paired_bootstrap=bootstrap, checkpoint_identity=identity,
 blocking_issues=[], nonblocking_issues=issues, claims=claims,
 limitations=[issue['detail'] for issue in issues] + [
     'All raw annotations and predictions were read numerically; no image content, neural model, training run or inference run was executed.',
     'The full1147-entry raw manifest and full protocol JSON were parsed and checked.70 supplied Python files were AST-parsed; metric, annotation, inference, policy, selection, controller and relevant vendor/config source paths were inspected directly.',
     'The large handoff is a historical append-only document. Its current overview and relevant claim sections were reviewed; exploratory full-document console reads were display-truncated and are recorded as such.',
     'Workspace bootstrap instruction files were read before task-specific inspection; their contents are not audit evidence. Prior audit verdicts mentioned by the supplied handoff were not adopted as proof.',
     'Complete numerical execution stdout/stderr/actual return codes are saved. Tool-level exploratory read metadata and retained outputs are saved; some early console output is only available in the host transcript and was display-truncated. No model/routing attestation or full host transcript export is available.'
 ])
write('EXPERIMENT_AUDIT.json', audit)

metric_rows = '\n'.join('| '+role+' | '+str(record['epoch'])+' | '+format(record['overall_metrics_percent']['PR'], '.12f')+' | '+format(record['overall_metrics_percent']['SR'], '.12f')+' |' for role, record in results['roles'].items())
interval_rows = '\n'.join('| '+metric+' | '+format(record['point_estimate'], '.12f')+' | ['+', '.join(format(x,'.12f') for x in record['percentile95'])+'] |' for metric, record in bootstrap['metrics'].items())
latency_rows = '\n'.join('| '+role+' | '+format(t['fps'],'.9f')+' | '+format(t['latency_mean_ms'],'.9f')+' | '+format(t['latency_p50_ms'],'.9f')+' | '+format(t['latency_p95_ms'],'.9f')+' |' for role,t in timing.items())
sections=[]
for key, value in checks.items():
    sections.append('## '+key.replace('_',' ')+' — '+value['status']+'\n\n'+value['details']+'\n\nEvidence:\n\n'+'\n'.join('- `'+e['path']+':'+str(e['line'])+'` — '+e['detail'] for e in value['evidence']))
issue_text='\n\n'.join('**'+issue['id']+' ('+issue['category']+')**. '+issue['detail']+' Action: '+issue['action'] for issue in issues)
claim_text='\n'.join('- **'+c['id']+' '+c['impact']+'** — '+c['claim']+' '+c.get('qualification','') for c in claims)
report=f'''# Experiment audit: R1 main versus pure CAD on RGBT210

**Overall verdict: WARN. Integrity status: warn. Blocking findings: 0.**

The supplied accuracy results are reproducible. All **{verification['total_checks']} named deterministic checks pass**, and all **{verification['input_files']} sealed inputs remain unchanged**. Both roles cover **210 sequences / 104,706 original frame pairs**, with 104,496 timed post-initialization frames each. Every supplied overall/per-sequence JSON/CSV score, mean curve, attribute aggregate and latency aggregate matches the independent recomputation exactly. There is no self-normalization, missing-sequence substitution or phantom score found in this pair.

Date: 2026-10-15. Fresh delegated reviewer: `{AGENT}`. Requested: `gpt-6-astra`, reasoning `max`, `fork_turns=none`. **Actual model/routing/reasoning/fork mode: UNATTESTED.** Review independence is **same-family**; acceptance status is **provisional**, never cross-family accepted. Deterministic arithmetic is accepted only for the supplied bytes and protocol.

Input root (read-only): `{INPUT}`. Output root: `{LOGICAL_OUT}`. Python is3.10.19 and NumPy2.2.6; exact interpreter and NumPy module path are in `runtime.json`. No neural import/execution, SSH/network, GPU query or package/environment change was performed.

| Role | Epoch | PR (%) | SR (%) |
|---|---:|---:|---:|
{metric_rows}

The paired5000 sequence bootstrap uses seed20261015, `numpy.random.default_rng` / PCG64, the saved toolkit sequence order, the same210 resampled named-sequence indices for PR and SR, and linear2.5/97.5% quantiles. Units below are percentage points, main minus pure.

| Metric | Paired point difference | Marginal95% percentile interval |
|---|---:|---:|
{interval_rows}

Both point estimates favor pure CAD, and both intervals include zero. This is neither evidence of superiority nor evidence of equivalence. It also does not establish significant inferiority at the displayed two-sided95% level. These are fixed-checkpoint sequence intervals: they exclude training-seed variation and selection uncertainty. Frames within a video are dependent; related events across videos may remain dependent. The two marginal95% intervals are not a simultaneous confidence region.

The protocol reads GT and predictions as float32, rounds prediction coordinates with `round(0)`, then replaces each first prediction with first GT. Despite the toolkit comment “ignore the first frame,” that frame remains in both numerator and denominator. Strict centers use `(w−1)/2+x`, and IoU uses inclusive pixel edges. PR counts error≤threshold at0…50 and reports threshold20; SR counts IoU>threshold at21 points0…1 and takes their arithmetic mean. No trapezoidal integration or model-dependent rescaling is introduced. All original annotations stay in the denominator, including the three nonpositive-width/height rows in car10, child4 and man24. The supplied scalar PR/SR definitions and the independent vector implementation agree at all30,240 sequence-curve values across both roles. The12 attributes are independently reconstructed from the supplied234-entry flag/name files and intersected with the210 benchmark names.

{chr(10).join(sections)}

## Checkpoint and completion evidence

Main `selected_checkpoint/best.pth` is2,603,720 bytes, SHA256 `{identity['sha256']}`. Its unique pickle epoch literal is10, matching selection, both receipts, report and controller state. Inspection used zip/pickletools without unpickling tensors or importing torch. The hash also matches the main controller declaration. All14 recorded source hashes and the configuration hash match the supplied files.

Pure checkpoint `/home/gaob/CADTrack_ep0014.pth.tar` has recorded collection-time size1,236,957,664 bytes and SHA256 `{identity['pure_checkpoint_receipt']['sha256']}`. The bytes are absent; epoch14 and250 loaded tensors are receipt declarations supported by the source loader, not independently loaded-state attestation. Main also depends on that external base checkpoint.

The four inference shard receipts each contain105 disjoint named sequences; shard frame totals are49,860 and54,846 for each role. Every per-sequence count and latency sum agrees. Recorded main/pure inference exits are0/0 and score exits0. Pure original stdout ends in exactly the supplied complete state; original stderr is empty. The observer recorded controller absence at04:25 UTC. These static facts do not independently attest an OS-level controller exit code or current external jobs. Training20 epochs /1600 updates and six-candidate DEV selection are internally consistent declarations; missing raw training and DEV traces are not reconstructed here. Main fit receipt `formal_metrics_complete=false` is a stage-local training receipt, superseded for this role by later completed selection/state/result artifacts.

## Timing

| Role | Recorded FPS | Mean latency (ms) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|
{latency_rows}

FPS is104,496 divided by the sum of recorded worker-frame seconds. It is not concurrent wall-clock throughput. Timers include decode/crop/visual forwards/decision/state or template update and ending CUDA synchronization; initialization and prediction serialization are excluded. Both supplied controller records describe two inference workers on physical GPU2, at different times. The raw reports’ “two-GPU” wording is stale. These measurements do not support matched speedup or lower complete-system cost.

## Nonblocking findings and actions

{issue_text}

The nonblocking classification permits qualified reporting of these already supplied numerical results. It does not waive the evidence needed for broader effectiveness or research-goal claims.

## Claim impact

{claim_text}

## Reproduction and forensic record

`verify_pair.py` independently reads the complete manifest/protocol, all210 GT files, both sets of210 prediction files and latency arrays, all four receipts and both reports. It then cross-checks numeric-only AST extracts of the supplied PR/SR source. There was one numerical verification execution, with actual return code0 and no failed numerical attempts. No mismatch was erased, no tolerance was changed, and no input was modified. The full manifest has1,147 entries; nine supplemental files are covered by the1,156-file initial seal. Seventy supplied Python files parse successfully. Direct semantic source inspection concentrates on the requested scoring, inference, selection, causal runtime and relevant vendor/config dependencies; it does not claim a line-by-line review of every historical handoff entry.

Artifacts: `deterministic_verification.json`; `recomputed_results.json`; `main_*` and `pure_*` per-sequence, curve, attribute and latency CSVs; `aggregate_latency_statistics.json`; `paired_bootstrap_5000.json`; `paired_bootstrap_5000_replicates.csv`; `paired_sequence_differences.csv`; `ground_truth_inventory.csv`; `checkpoint_identity_verification.json`; `input_hashes_before.json` / `input_hashes_after.json`; `execution/001_verify_pair.*`; `source_read_tool_trace*.json`; the request/response trace under `.aris/traces/experiment-audit/2026-10-15_run01/`. Final validation and output SHA manifest are separate artifacts produced after report writing.

Large exploratory documentation outputs were display-truncated and this limitation is retained in the tool trace. All numerical inputs and the entire raw manifest/protocol were nevertheless read and checked in full by the verifier. Early exploratory tool outputs remain partly in the host transcript; this audit cannot export a complete host transcript or attest host routing. No raw GT/prediction coordinates, checkpoint contents or per-frame latency arrays are emitted into the audit reports.
'''
# These prose-only replacements keep the audit's long-form numbers readable.
for old, new in {
    'All4154': 'All 4,154', 'all210': 'all 210', 'all1147': 'all 1,147',
    'same210': 'same 210', 'both210': 'both 210', 'the210': 'the 210', 'the12': 'the 12',
    'All14': 'All 14', 'all30,240': 'all 30,240', 'full1147': 'full 1,147',
    'paired5000': 'paired 5,000', 'seed20261015': 'seed 20261015', 'seed42': 'seed 42',
    'epoch10': 'epoch 10', 'epoch14': 'epoch 14', '20-epoch': '20-epoch',
    'is3.10.19': 'is 3.10.19', 'NumPy2.2.6': 'NumPy 2.2.6',
    'linear2.5': 'linear 2.5', 'Marginal95%': 'Marginal 95%', 'marginal95%': 'marginal 95%',
    'two-sided95%': 'two-sided 95%', 'both95%': 'both 95%', 'simultaneous95%': 'simultaneous 95%',
    'thresholds0': 'thresholds 0', 'at0': 'at 0', 'at21': 'at 21', 'points0': 'points 0',
    'threshold20': 'threshold 20', 'the21': 'the 21',
    'supplied234': 'supplied 234', 'is2,603,720': 'is 2,603,720', 'is10': 'is 10',
    'size1,236,957,664': 'size 1,236,957,664', 'and250': 'and 250',
    'contain105': 'contain 105', 'are49,860': 'are 49,860', 'and54,846': 'and 54,846',
    'exits0': 'exits 0', 'are0/0': 'are 0/0', 'at04:25': 'at 04:25',
    'Training20': 'Training 20', '/1600': '/ 1,600',
    'FPS is104,496': 'FPS is 104,496', 'code0': 'code 0',
    'has1,147': 'has 1,147', 'the1,156': 'the 1,156',
    'roleby': 'role by', 'and12': 'and 12', 'all four receipts': 'all four receipts'
}.items():
    report = report.replace(old, new)
(OUT / 'EXPERIMENT_AUDIT.md').write_text(report, encoding='utf-8')

trace = '.aris/traces/experiment-audit/2026-10-15_run01'
request = (OUT / 'review_request.txt').read_text(encoding='utf-8')
write(trace+'/run.meta.json', dict(skill='experiment-audit', run_id='2026-10-15_run01', date='2026-10-15',
    agent_id=AGENT, requested_model='gpt-6-astra', requested_reasoning_effort='max', requested_fork_turns='none',
    actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED', actual_routing='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional', input_root=INPUT, output_root=LOGICAL_OUT))
write(trace+'/001-pair-audit.request.json', dict(call_number=1, purpose='fresh-completed-RGBT210-pair-integrity-audit',
    tool='collaboration.spawn_agent (requested route; parent host call not independently attested here)',
    requested_model='gpt-6-astra', requested_reasoning_effort='max', requested_fork_turns='none', prompt=request,
    request_sha256=hashlib.sha256(request.encode()).hexdigest(), request_path=LOGICAL_OUT+'/review_request.txt'))
(OUT / trace / '001-pair-audit.response.md').write_text(report, encoding='utf-8')
write(trace+'/001-pair-audit.meta.json', dict(call_number=1, purpose='fresh-completed-RGBT210-pair-integrity-audit',
    agent_id=AGENT, generated_at=now, status='completed_WARN', verdict='WARN', deterministic_status='PASS',
    actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED', actual_routing='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    blocking_issues=0, output_response=LOGICAL_OUT+'/'+trace+'/001-pair-audit.response.md'))
write('source_and_claim_review.json', dict(checks=checks, nonblocking_issues=issues, claims=claims))
print(json.dumps(dict(verdict='WARN', blocking_issues=0, nonblocking_issues=len(issues), report=LOGICAL_OUT+'/EXPERIMENT_AUDIT.md',
    machine_report=LOGICAL_OUT+'/EXPERIMENT_AUDIT.json', deterministic_checks=verification['total_checks'])))
