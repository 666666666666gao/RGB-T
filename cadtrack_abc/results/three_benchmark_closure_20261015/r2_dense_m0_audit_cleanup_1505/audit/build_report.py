"""Assemble the semantic report and private review trace from completed evidence."""
import datetime, hashlib, json
from pathlib import Path
OUT=Path(__file__).resolve().parent
INPUT=OUT.parent/'dense_m0_fresh_audit_inputs_1435'
M=json.loads((INPUT/'input_manifest.json').read_text(encoding='utf-8-sig'))
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
parts={name:read(file) for name,file in [('artifacts','artifact_checks.json'),('source_stdlib','source_stdlib_checks.json'),('orchestration','orchestration_checks.json'),('lineage_coverage','lineage_coverage_checks.json')]}
count=sum(r['checks_count'] for r in parts.values())
assert count==848 and all(r['status']=='PASS' for r in parts.values())
before=read('input_seal_before.json'); after=read('input_seal_after.json')
assert before['files']==after['files'] and before['manifest_sha256']==after['manifest_sha256'] and before['index_sha256']==after['index_sha256']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
B='cadtrack_abc/results/three_benchmark_closure_20261015/'
A=B+'r2_dense_admission_1425/'
evidence={
 'A':['cadtrack_abc/data.py:73','cadtrack_abc/data.py:82','cadtrack_abc/dense_model.py:180','cadtrack_abc/dense_model.py:202','cadtrack_abc/dense_model.py:233','cadtrack_abc/dense_model.py:250','cadtrack_abc/dense_tracker.py:36','cadtrack_abc/dense_evaluation.py:18','cadtrack_abc/validate_dense.py:39','cadtrack_abc/infer_dense.py:34','cadtrack_abc/score.py:35'],
 'B':['cadtrack_abc/data.py:51','cadtrack_abc/mechanism_summary.py:8','cadtrack_abc/mechanism_summary.py:36','cadtrack_abc/mechanism_summary.py:102','cadtrack_abc/score.py:54','cadtrack_abc/score_dense.py:40','cadtrack_abc/select_dense.py:46'],
 'C':[A+'admission.json:4',A+'batch128/HA_batch128/fit_complete.json:5',A+'batch128/HA_batch128/progress.json:18',A+'batch128/HA_batch128/history.json:7',B+'r2_dense_dependency_fix_1405/first_attempt_observation.json:31',B+'r2_dense_dependency_fix_1405/source_manifest.json:3',B+'r2_dense_full_launch_1430/launch_verified.json:75',B+'r2_dense_plan_20261015.json:2','docs/HANDOFF_20261002.md:5376'],
 'D':['cadtrack_abc/train_dense.py:116','cadtrack_abc/dense_model.py:251','cadtrack_abc/dense_model.py:267','cadtrack_abc/score.py:55','cadtrack_abc/score.py:93','cadtrack_abc/dense_pipeline.py:73','cadtrack_abc/dense_pipeline.py:82','cadtrack_abc/dense_pipeline.py:104',A+'dense_full_pair_driver_remote.py:50'],
 'E':['cadtrack_abc/qualify_dense.py:35','cadtrack_abc/qualify_dense.py:73',A+'admission.json:133','cadtrack_abc/train_dense.py:55','cadtrack_abc/train_dense.py:78','cadtrack_abc/train_dense.py:113','cadtrack_abc/select_dense.py:49','cadtrack_abc/select_dense.py:66',B+'r2_dense_plan_20261015.json:38','docs/HANDOFF_20261002.md:5365'],
 'F':[B+'r2_dense_dependency_fix_1405/dependency_cpu_check.json:17','cadtrack_abc/qualify_dense.py:52','cadtrack_abc/qualify_dense.py:73','cadtrack_abc/validate_dense.py:58','cadtrack_abc/score.py:35','cadtrack_abc/score_dense.py:32']}
details={
 'A':('WARN','Dataset annotations are the intended loss/evaluation truth; later labels do not enter the online tracker. Parity deliberately compares model outputs and is a labeled qualification proxy. Dataset bytes and official toolkit internals are absent, so source causality is supported but dataset origin/authenticity is not independently attested.'),
 'B':('PASS','Inspected IoU/diagnostic denominators are box union, valid frames, events or time. Official outputs are multiplied by100 and deltas are percentage-point subtraction. No prediction-max/min/mean performance-score rescaling found in supplied source. Official toolkit implementation is outside this seal.'),
 'C':('WARN','Qualification counts, source hashes, gradient/freeze/reload receipt fields, parameter sums and timing arithmetic agree. First failed import/exit1 is retained. Full-run launch is a historical process-start snapshot; dense full-training, selection and formal reports are absent. Stale plan purpose/header and launch-time250-tensor wording need qualification.'),
 'D':('WARN','Losses, parity comparison, native scorer dispatch, IoU selection and mechanism summaries have reachable call sites. Full dense selection/inference/scoring source is present but not exercised by completed artifacts in this input. R1 B/C code is not executed by dense H/HA. No active fatal dead-code defect found.'),
 'E':('WARN','Evidence covers one seed, six two-update probes and24 parity frames. It supports limited implementation/qualification claims, not tracking gains, all881 coverage, complete20 epochs, B/C effectiveness or seven-metric success. Cold-start throughput, repeated development selection and unknown pretrained development exposure must remain explicit.'),
 'F':('PASS','Each evidence type is distinguished: model-to-model parity is synthetic_proxy on real images; CPU synthetic tensors are simulation_only; training loss supervision and intended full98/formal scoring are real_gt. Qualification and initialization equality are not tracking-accuracy results.')}
warnings=[
 dict(id='W1',kind='evidence_limit',blocking=False,title='Runtime qualification is receipt-supported, not independently re-executed',evidence=evidence['C'][:6],details='No weights, images, annotations, raw order streams, complete optimizer logs or hardware authentication supplied. Aggregate per-group gradients and change counts do not prove useful learning or every mechanism effective. The86,045,184 frozen count lacks its per-tensor inventory; do not label it independently recomputed.'),
 dict(id='W2',kind='pending_runtime_obligation',blocking=False,title='Full training, full98 selection and three formal datasets remain unclosed in this snapshot',evidence=['cadtrack_abc/dense_pipeline.py:74','cadtrack_abc/select_dense.py:23',B+'r2_dense_full_launch_1430/launch_verified.json:89'],details='The launch verifies process command lines at06:27:57UTC, not optimizer progress, completion, current live status or performance. Source and synthetic control-flow tests do not discharge terminal evidence obligations.'),
 dict(id='W3',kind='claim_limit',blocking=False,title='Initial equivalence and supervised qualification are not tracking gains',evidence=['cadtrack_abc/qualify_dense.py:52','cadtrack_abc/qualify_dense.py:73','docs/HANDOFF_20261002.md:5351'],details='24 query frames on2 TRAIN clips contain4 reference template-update events and exact outputs, not a GT-quality evaluation or whole-distribution equivalence proof.'),
 dict(id='W4',kind='scope_limit',blocking=False,title='One seed; dense H/HA only; repeated development selection',evidence=['cadtrack_abc/infer_dense.py:51',B+'r2_dense_plan_20261015.json:38','cadtrack_abc/select_dense.py:49'],details='Do not infer B/C, allABC, module causality, stable seeds, untouched DEV, or positive training gain from a positive-epoch checkpoint. The protocol explicitly permits a worse trained-best while retaining an epoch0 overall best.'),
 dict(id='W5',kind='documentation_drift',blocking=False,title='Stale current-state summaries and unsupported launch-time strict-load wording',evidence=['docs/HANDOFF_20261002.md:1','docs/HANDOFF_20261002.md:5376',B+'r2_dense_plan_20261015.json:2',B+'r2_dense_plan_20261015.json:4',B+'r2_dense_full_launch_1430/launch_verified.json:59'],details='Plan status says running but purpose says not yet launched; handoff heading is older R1 status. Launch receipt includes pretrained SHA and process commands but no fresh250-tensor strict-load record. That250 value is supported for M0; full-run startup requires its manifest/log. Refresh current summary without rewriting chronological historical receipts.'),
 dict(id='W6',kind='provenance_limit',blocking=False,title='Official toolkit and baseline/pretraining semantic provenance are outside the seal',evidence=['cadtrack_abc/score.py:20','cadtrack_abc/score_dense.py:31','cadtrack_abc/baseline.py:27','cadtrack_abc/environment.json:13'],details='22 launch source hashes and12 M0 source hashes match included bytes, but the set is not a complete dependency closure. External rgbt implementation, full upstream import graph, original baseline weights and prior RGB baseline reports are missing. Historical allABC/DDP witness text in environment.json is not a dense H/HA witness.'),
 dict(id='W7',kind='measurement_limit',blocking=False,title='Batch recommendation is based on two-update cold-start throughput/allocation',evidence=[A+'batch32_64/driver.py:71',A+'admission.json:133','cadtrack_abc/train_dense.py:108','cadtrack_abc/train_dense.py:141'],details='Batch128 wins only among32/64/128 tested. Peak allocated bytes are not total process/reserved/device memory. No GPU identity/model attestation or steady-state efficiency replicate supplied; no inference speed or training completion ETA is inferred.')]
obligations=[
 'Preserve actual full-run20-epoch terminal receipts per variant:400000 clips,3200000 query observations,3140 optimizer updates, all20 order hashes, gradient/freeze/strict-reload evidence and true process exits.',
 'Complete all six98-video candidates per variant with exact sequence inventory,52200 frames/snapshot, checkpoint hashes and strict-load receipts; retain overall best and best positive epoch separately.',
 'Carry the same preselected positive-epoch checkpoint/hash/configuration through245/220703 LasHeR,234/116649 RGBT234 and210/104706 RGBT210, including negative results. Archive real per-sequence metrics, curves, attributes and provenance, then independently recompute against actual dataset GT/toolkit.',
 'Close the pair only after both rows and all20 actual clip-order hashes match. Independent data RNG in source and three probe hash matches are not substitutes for final order evidence.',
 'Keep baseline LasHeR scalar-reference limitation, no paired uncertainty from unavailable raw baselines, repeatedDEV/pretraining exposure limitation, single-seed scope, head-training control and H/HA-only scope visible.',
 'Refresh plan purpose/current handoff header and limit the full-launch250-tensor statement to evidence actually available. Do not recast archived earlier statuses as current live state.',
 'No new GPU gate, restart, requalification or checkpoint modification is required by this audit. Existing work may continue; the above are claim-closure obligations.']
claims=[
 dict(id='qualification_counts_and_arithmetic',impact='supported',limit='Within provided closed receipts:12 updates,896 clip observations,7168 query-frame observations; no raw-data or weight authentication.'),
 dict(id='initial_parity',impact='supported_with_qualifier',limit='24 query frames on2 TRAIN clips, H/HA output/confidence/template values+strides equal pure reference; no accuracy claim.'),
 dict(id='data_order',impact='supported_with_qualifier',limit='Separate generator source;3 recorded H/HA hash pairs equal. Actual order cannot be reconstructed without streams/Torch/data and future20 epochs remain pending.'),
 dict(id='trainable_parameters',impact='supported',limit='H17093253; HA17500424; added407171 from both manifest names and independent layer dimensions.'),
 dict(id='frozen_backbone_and_reload',impact='receipt_supported',limit='Before/after hashes and strict reload booleans agree with reachable source assertions; full tensors not supplied, frozen parameter count not independently rebuilt.'),
 dict(id='cold_start_batch_choice',impact='supported_with_qualifier',limit='128 gmean3.7079203378707413 clips/s; relative +15.3021117862% vs64,+41.6151921300% vs32; not steady-state method speed.'),
 dict(id='full_training_launch',impact='snapshot_supported',limit='Two intended full training command lines observed at2026-10-15T06:27:57.283538Z. No present-live-state claim or completed updates implied.'),
 dict(id='formal_performance_or_plus_two',impact='unsupported_now',limit='No completed dense full-training/selection/formal result artifacts in seal; not a fabricated claim where narrative explicitly leaves it pending.'),
 dict(id='C1_dense_history_multifuture_gain',impact='unsupported_now',limit='A insertion is implemented and qualified; multifuture B, equal-capacity controls and tracking gain not established.'),
 dict(id='C2_actual_consequence_search_and_submission',impact='unsupported_now',limit='B/C not integrated in current dense H/HA path; no fullABC effectiveness claim permitted.')]
report={
 'audit_skill':'experiment-audit','date':'2026-10-15','generated_at':now,'verdict':'WARN','overall_verdict':'WARN','integrity_status':'warn',
 'blocking_findings_count':0,'blocking_findings':[],'reason_code':'qualification_supported_formal_completion_and_provenance_limited',
 'summary':'Closed M0 receipts and current dense source are internally consistent. No blocking active source/integrity defect identified. Full training/selection/formal performance is not established by this immutable snapshot.',
 'requested_reviewer_model':'gpt-6-astra','requested_reviewer_reasoning_effort':'max','actual_reviewer_model':'UNATTESTED','actual_reviewer_routing':'UNATTESTED','actual_reviewer_effort':'UNATTESTED',
 'reviewer_family':'openai','review_independence':'same-family','acceptance_status':'provisional','agent_id':'/root/dense_m0_integrity_1435',
 'executor_model':'UNATTESTED','reviewer_model':'UNATTESTED','reviewer_reasoning':'UNATTESTED','reviewer_request_satisfaction':'UNATTESTED; requested routing is not independent evidence of actual routing',
 'scope':{'input_manifest':str(INPUT/'input_manifest.json'),'input_count':85,'input_bytes':before['bytes'],
          'source_files':41,'json_files':42,'yaml_files':1,'narrative_files':1,'json_leaf_count':8059,
          'runtime_boundary':'Closed M0 plus full-launch historical snapshot only; no current-state polling',
          'semantic_coverage':'All supplied source reviewed line by line; all JSON parsed/traversed. Entire historical handoff read as bytes/headings, current dense/R2 section5242-5380 read in full; older independent research claims are not recertified.',
          'not_provided':['images','dataset annotation bytes','model/checkpoint tensors','raw per-update order stream','complete original training logs','official rgbt toolkit source','complete native import graph','prior baseline formal reports']},
 'checks':{letter:{'status':details[letter][0],'details':details[letter][1],'evidence':evidence[letter]} for letter in 'ABCDEF'},
 'warnings':warnings,'obligations':obligations,'claims':claims,
 'evaluation_types':[
  {'artifact':'M0 epoch0 parity','type':'synthetic_proxy','reference':'PureCADTracker model output on2 real TRAIN clips','completed_receipt_present':True,'accuracy_claim_allowed':False},
  {'artifact':'CPU shape/import prerequisite','type':'simulation_only','reference':'Synthetic random tensors; mathematical equality and gradient predicates','completed_receipt_present':True,'accuracy_claim_allowed':False},
  {'artifact':'Six finite training probes','type':'real_gt','reference':'Dataset init.txt for supervised training losses, not held-out accuracy evaluation','completed_receipt_present':True,'accuracy_claim_allowed':False},
  {'artifact':'Planned repeated full98 development','type':'real_gt','reference':'Dataset init.txt, valid positive-size boxes, first frame included, sequence mean IoU not official SR','completed_receipt_present':False},
  {'artifact':'Planned formal three datasets','type':'real_gt','reference':'Dataset-specific native rgbt evaluator; no prediction-derived truth intended','completed_receipt_present':False},
  {'artifact':'This audit control-flow simulation','type':'simulation_only','reference':'In-memory fabricated receipts used to check source contracts','accuracy_claim_allowed':False}],
 'deterministic_verification':{'counted_assertions':count,'failed_counted_assertions':0,'checker_counts':{k:r['checks_count'] for k,r in parts.items()},
   'input_seals_before_and_after':{'files_each':85,'all_match':True,'manifest_sha256':before['manifest_sha256'],'path_index_sha256':before['index_sha256']},
   'execution':'CPU stdlib only. NumPy import attempt failed before executing any tests; no package installed.',
   'retained_failure_kinds':['GBK display UnicodeEncodeError; initial merged tool event and separately captured reproduction','NumPy ModuleNotFoundError; exact failed script/stdout/stderr/exit retained','Audit runner argparse forwarding mistake; original tool event and captured reproduction retained'],
   'synthetic_controlflow_scenarios':5,'actual_neural_executions_in_audit':0},
 'audited_input_hashes':{f['relative']:'sha256:'+f['sha256'] for f in M['files']},
 'trace_path':'.aris/traces/experiment-audit/2026-10-15_dense_m0_1435/',
 'trace_limit':'Private request and full substantive report preserved. Complete host/collaboration transcript export unavailable; early tool output displays included truncation. No fabricated full-host trace.',
 'output_hash_manifest':'output_manifest.json','output_hash_manifest_digest_file':'output_manifest.sha256',
 'audit_constraints':{'input_or_project_edits':False,'network_or_ssh':False,'torch_cuda_or_neural_execution':False,'gpu_resource_process_polling':False,'package_installs':False,'git_mutations':False,'subagents_spawned':False,'external_messages':False,'training_restarted_or_gated':False}}
(OUT/'EXPERIMENT_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

md='''# Experiment Audit — dense R2 M0 and launch snapshot

**Verdict: WARN. Blocking findings: 0.** The sealed M0 receipts, numerical summaries and current source agree. No active source defect was found that requires interrupting or restarting the existing training. This is a qualification/source audit, not acceptance of full training or tracking performance.

Requested reviewer: **gpt-6-astra / max**. Actual model, routing and effort: **UNATTESTED**. Fresh reviewer context; `review_independence: same-family`; `acceptance_status: provisional`. Requested routing is not proof of actual routing.

## Evidence boundary and verification

All references below are relative to the sealed `project/` beside `input_manifest.json`. `R/` means `cadtrack_abc/results/three_benchmark_closure_20261015/`; `M0/` means `R/r2_dense_admission_1425/`. The JSON report carries full relative evidence paths.

85 files (1,347,440 bytes):41 Python,42 JSON,1 YAML,1 historical narrative. All input hashes and sizes matched before and after; the path index matches the manifest. All supplied sources were inspected line by line. All JSON was parsed and8,059 leaves traversed, including complete881/98 name lists. The899,460-byte historical handoff was read as bytes and fully indexed by headings; the current dense/R2 section5242–5380 was read in full. Older GOLA/R1 result claims are not independently recertified by this scope. No missing input is silently assumed verified.

**848 counted assertions passed**:684 artifact/arithmetic checks,40 source/stdlib checks,38 orchestration checks,86 lineage/coverage checks. The input before/after seals are additionally retained. Exact scripts, stdout, stderr, exits and script hashes are under `runs/`; `artifact_checks.json`, `source_stdlib_checks.json`, `orchestration_checks.json`, and `lineage_coverage_checks.json` enumerate the checks. The orchestration tests use in-memory fabricated phase receipts and never execute a neural phase.

Three checker infrastructure failure kinds are preserved: Windows GBK display encoding, an unavailable NumPy import, and runner argument forwarding. The original failed NumPy source and each captured reproduction remain present. The final substantive checkers use stdlib only. No Torch, CUDA, neural work, installs, network/SSH, process/resource polling, Git mutation or project/input edit occurred. Complete host transcript export was unavailable; this trace does not claim to contain one.

## A. Ground truth provenance — WARN

Training `VideoClips` reads dataset `init.txt` and paired visible/infrared frames (`cadtrack_abc/data.py:71`, `:73`, `:88`, `:92`). A valid initial annotation determines a sampled clip start. Later boxes are supervision: `DenseCADTrack.step_batch(images,state)` crops around previous predicted state (`dense_model.py:180`), then the loss reads current truth (`:202`, `:251`, `:254`). The memory teacher is detached causal past evidence, filtered using past training GT, and is passed only to the loss (`:233`, `:257`, `:260`). It is not an inference memory oracle. GT-selected regression-cell indexing (`:212`) is supervised learning, not a GT-driven tracking choice.

The inference tracker accepts only the next image (`dense_tracker.py:36`). `track_video` initializes from one box and then calls `track` (`dense_evaluation.py:18`); formal inference reads just the first annotation line (`infer_dense.py:34`). Development scoring independently loads all dataset labels and computes IoU after tracking (`validate_dense.py:39`). Official scoring sets ground truth from dataset files and invokes the dataset-specific evaluator (`score.py:35`, `:39`, `:55`). No baseline output is used as accuracy ground truth in these paths.

The epoch0 comparison explicitly uses `PureCADTracker` predictions as its reference (`qualify_dense.py:52`) and explicitly calls itself a two-TRAIN-clip qualification, not accuracy (`:73`). This is a disclosed output-consistency proxy. Actual images, annotation bytes and external toolkit source were not exported. The audit supports the source causality and declared GT route; it cannot authenticate dataset identity, annotation contents or external official metric internals from this seal alone.

## B. Score normalization — PASS within supplied source

IoU divides intersection by box union (`data.py:51`, `mechanism_summary.py:8`); diagnostics divide by valid frames, counted writes/events or elapsed time (`mechanism_summary.py:36`, `:102`). Candidate scores/Hann weights, feature normalization, attention softmax and GT heatmap normalization are internal model/loss operations, not performance-score inflation. Selection uses valid-box mean IoU (`select_dense.py:46`); official scores are multiplied by100 and baseline deltas are subtracted in percentage points (`score.py:54`, `score_dense.py:40`). No performance score divided by its own prediction max/min/mean was found.

The native `rgbt` library is external (`score.py:20`) and absent from the seal, so its exact PR/NPR/SR/MPR/MSR implementation is not certified here. Event-conditioned diagnostic rates must retain their explicit denominators. First-frame-inclusive development IoU is not official SR and must remain labeled as such (`select_dense.py:66`).

## C. Result existence, arithmetic and provenance — WARN

Closed receipts support six finite probes: H/HA at batches32,64,128, **2 updates each**,12 total. They contain896 raw clip observations and7,168 query-frame observations. These are repeated observations, not896 unique clips or all881 training videos; the largest probe samples256 dataset indices. Each terminal receipt states `probe:true`, one epoch, no full98 selection and null formal metrics (for example `M0/batch128/HA_batch128/fit_complete.json:5`, `:30`). Manifest `QUALIFICATION_RUNNING` is a creation-time status; later terminal/exit receipts establish completion, not that stale field.

The24 parity rows are two sequences with12 contiguous query frames each, all H/HA box errors0 and all output/template comparisons true;4 reference template-update events occur. This verifies a finite behavior sample, not tracking quality (`qualify_dense.py:35`, `:52`; `M0/batch32_64/epoch0_parity.json`).

H has17,093,253 trainable parameters; HA17,500,424; added prompt407,171. Both sums were independently recomputed from manifest entries and layer dimensions (`M0/batch128/HA_batch128/manifest.json:1016`; `dense_model.py:48`; `ported_modules.py:74`; `vendor/CADTrack/lib/models/layers/head.py:97`). Recorded gradients for fusion/head and enabledA are positive and finite; changed-tensor counts are4/54/(11). The source exposes differentiable fusion/head outside frozen extraction and asserts no backbone gradients, a fixed backbone hash and strict dense reload (`dense_model.py:115`, `:132`, `:148`; `train_dense.py:119`, `:159`, `:167`). Terminal receipts agree (`M0/batch128/HA_batch128/progress.json:18`, `fit_complete.json:21`). These are source-and-receipt-supported runtime claims: absent tensors prevent independent gradient/reload/freeze reproduction. The86,045,184 frozen-parameter count has no full backbone tensor inventory in this input and was not independently rebuilt. A positive group norm/change count is not evidence that every mechanism learned usefully.

All12 hashes in the M0 source manifest and22 expected full-launch source hashes match sealed bytes. The additional21-source preflights and20-source initial-failure preflight also match. Embedded full driver/wrapper source text agrees with archived scripts. M0 state→source-manifest, capacity→base-state, admission→pipeline and full-launch→capacity-state hash links agree. These verify internal lineage, not third-party authenticity or a complete transitive dependency closure.

Chronology is coherent: first parity import fails at06:00:37UTC (exit1, missing `lib.utils.tensor`); retry closes06:11:06; batch128 closes06:19:56; full training launch is recorded06:27:52 and process commands observed06:27:57. The original failure is retained (`R/r2_dense_dependency_fix_1405/first_attempt_observation.json:31`, `:43`), and the required native dependency is included (`vendor/CADTrack/lib/utils/__init__.py:1`). The current audit did not rerun that prerequisite or poll these PIDs.

**No dense full-training terminal, completed full98 selection, formal prediction/report or pair-completion artifact is included.** The launch receipt records intended full command lines, not completed optimization or live present status (`R/r2_dense_full_launch_1430/launch_verified.json:75`). The latest handoff correctly leaves formal updates/results pending (`docs/HANDOFF_20261002.md:5378`). Nonblocking documentation drift remains: the plan's status says running but its purpose says not yet launched (`R/r2_dense_plan_20261015.json:2`, `:4`); the handoff title is older R1 status (`docs/HANDOFF_20261002.md:1`). Its full-launch250-tensor wording (`:5376`) is not established by the launch snapshot:250 is evidenced for M0, while a full-run strict-load manifest/log is absent.

## D. Called versus dead code; orchestration — WARN

The actual training call reaches dense forward/loss/backward/optimizer (`train_dense.py:116`). Dense localization and enabled memory losses are called from forward (`dense_model.py:254`, `:257`), with loss parts present in probe receipts. Parity comparison is called in its loop and has matching rows. Official scorer dispatch and mechanism summaries are reachable (`score.py:55`, `:93`), but their new dense outputs are not yet evidenced by completed files. `sequence_iou` returned by dense forward is computed but unused by this trainer; it is not a reported training-monitor score. Full-video validation recomputes its own IoU. Imported older R1 motion/action code and unused native utilities are not proof that B/C runs in dense H/HA.

The new pipeline explicitly requires20 epochs,400,000 clips,3,200,000 query observations,3,140 updates atbatch128, successful reload/freeze and candidate epochs0/1/2/5/10/20 before selection (`dense_pipeline.py:74`). It dispatches two98-video shards per candidate (`:82`), selects positive-epoch `trained_best` (`:93`), and sends that same checkpoint to all three datasets with digest/count checks (`:104`, `:110`, `:119`, `:125`). The pair driver finally compares all20 actual per-epoch order hashes and collects both full result rows (`M0/dense_full_pair_driver_remote.py:50`).

A stdlib simulation executed the actual pipeline main with only the nested external-phase runner replaced and an in-memory filesystem: two normal H/HA paths, plus rejected undertraining, epoch0-as-trained-best and changed-checkpoint cases. Both normal paths carried negative fabricated scores through all datasets; each created14 ordered phase calls/23 child-command specifications, at most two commands per card phase. Actual paired-order assertions accepted matching20-row histories and rejected a last-epoch mismatch. These checks validate orchestration logic only. Neural execution, toolkit outputs, disk behavior under the real run and true completion remain untested here. No active fatal dead-code or orchestration defect was established.

## E. Scope, seeds, selection and efficiency — WARN

The observed scope is seed42, two variants, three batches, two updates each, and two TRAIN parity clips. The881/98 split is disjoint, identical across all six manifests and exactly reproduced from split seed20261013. The loader receives a separate generator42 (`train_dense.py:78`) and hashes actual sequence plus initial/query frame order (`:113`). All three recorded paired hashes match. This checks the provided hashes; without raw streams/data/Torch it does not independently reconstruct the actual order. Final20-epoch pairing is still an obligation.

Per-run cold-start throughput recomputes as clips/epoch seconds (`M0/batch32_64/driver.py:71`):

| Batch | H clips/s | HA clips/s | Geometric mean | H peak allocated GiB | HA peak allocated GiB |
|---|---:|---:|---:|---:|---:|
|32|2.6273026993|2.6093418892|2.6183068936|2.6692457199|3.7064023018|
|64|3.2930638281|3.1404083136|3.2158303785|4.8040499687|6.8711657524|
|128|3.8267103732|3.5928178229|3.7079203379|9.0770988464|13.1935615540|

128 is15.3021117862% above64 and41.6151921300% above32 by geometric mean. Its H/HA allocations equal37.8212%/54.9732% of the reported24GiB capacity. The record is peak **allocated** memory, not total/reserved/driver memory; GPU model identity is not attested. These two-update measurements include cold data loading/optimizer initialization and worker completion, and do not establish steady-state throughput, globally optimal batch, inference speed or a reliable full-run ETA (`M0/admission.json:133`).

Full98 selection uses repeated development videos, sequence-equal valid-box IoU including the first initialized frame, not official SR (`validate_dense.py:58`, `select_dense.py:66`). The supplied epoch14 pretraining's possible DEV exposure is explicitly unattested (`R/r2_dense_plan_20261015.json:38`). Overall best and positive-epoch best are independently selected (`select_dense.py:49`): a positive-epoch winner can be worse than initialization. It must still be reported, and is not automatically a training gain. Both variants train fusion/head BatchNorm; this is disclosed and is not proven identical to the original paper trainer (`train_dense.py:66`; native YAML:77).

Current H/HA tests do not establish persistent identity tracklets, B multifuture search, C consequence-aware submission, completeABC, independent module contributions, multiseed stability or seven-metric+2 success. Single-seed scope is a claim limit, not a new demand for three seeds. The plan labels these later obligations and current goal unmet; no new experiment gate is imposed.

## F. Evaluation type — PASS

| Artifact/path | Classification | Claim ceiling |
|---|---|---|
|Epoch0 pure/H/HA comparison|`synthetic_proxy` on real TRAIN imagery|Finite model-output consistency only|
|CPU tensor/import prerequisite|`simulation_only`|Synthetic shape/equality/gradient prerequisite only|
|Six optimization probes|`real_gt` training supervision|Finite optimizer/source qualification; no held-out accuracy|
|Planned full98 development|`real_gt`|Repeated development IoU, initialization included; completed artifacts absent|
|Planned three formal benchmarks|`real_gt` intended|Dataset-specific official metrics only after real outputs/recomputation|
|This audit's in-memory fixtures|`simulation_only`|Source control-flow witness, never a training/result receipt|

The classification is supported by explicit scope text in `qualify_dense.py:73`, `R/r2_dense_dependency_fix_1405/dependency_cpu_check.json:17`, `validate_dense.py:58`, and `score_dense.py:32`. Timing/resource measurements are auxiliary measurements, not a GT accuracy evaluation.

## Obligations and claim impact

1. Preserve real20-epoch fit terminals, full candidate manifests, actual exits, all20 paired order hashes and full-run freeze/reload/gradient evidence. Source presence and launch commands do not replace them.
2. Close six full98 candidates per variant with98 identities/52,200 frames each; preserve both overall and positive best. Complete the same positive checkpoint across245/220,703,234/116,649 and210/104,706 sequences/frames and retain negative results.
3. Before performance claims, retain actual dataset/toolkit provenance, full per-sequence/attribute/curve results and same-checkpoint evidence; recompute the metrics. LasHeR scalar references do not permit paired baseline uncertainty. Missing historical baseline/weight provenance stays explicit.
4. Describe current work as frozen-ViT H versus HA localization/prompt training, seed42, with repeatedDEV selection and unknown original DEV exposure. C1/C2 in the plan remain hypotheses, not achieved findings.
5. Refresh stale current summaries and qualify the full-launch250-tensor wording. Preserve chronological failed/pending receipts as historical facts.

**None of these findings requests a new GPU qualification, restart or interruption.** They delimit what can be claimed and what later completion receipts must establish. Exact output hashes are in `output_manifest.json` with detached digest `output_manifest.sha256`; private leaf inventory and `.aris` traces must remain private.
'''
(OUT/'EXPERIMENT_AUDIT.md').write_text(md,encoding='utf-8')

trace=OUT/'.aris/traces/experiment-audit/2026-10-15_dense_m0_1435'
trace.mkdir(parents=True,exist_ok=True)
prompt='''Use the experiment-audit skill at C:/Users/gb/.codex/skills/experiment-audit/SKILL.md. You are a fresh adversarial experiment-integrity reviewer. Trust no author interpretation; read every listed artifact directly. Inputs: C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/dense_m0_fresh_audit_inputs_1435/input_manifest.json (lists 85 sealed source/result/config/narrative files under project/); path index C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/dense_m0_fresh_audit_1435_paths.json. Output ONLY to new directory C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/dense_m0_fresh_audit_20261015_1435, including EXPERIMENT_AUDIT.md and EXPERIMENT_AUDIT.json plus reproducible CPU checkers, actual stdout/stderr/exits and hashes. Do not modify inputs or current project files. Do not read personal MEMORY/USER/SOUL files. Do not spawn additional agents. No SSH/network, Torch/CUDA/neural execution, GPU/resource/process/status polling, training/checkpoint loading, package installs, Git mutations or external messages. Local CPU stdlib/AST/NumPy deterministic checks only; existing E:/python.exe may be used. Review actual source line by line, artifacts, source/receipt provenance, and chronology. Audit checklist A-F from the skill: dataset GT versus output/synthetic references; prediction-dependent normalization; exact numeric/report existence and real execution evidence; called versus dead code; scope/seed/training/evaluation claims; evaluation-type classification. Specifically examine data/label causality, initial equality versus tracking quality, source gradient and freeze/reload claims versus available receipts, finite optimizer counts versus full prescribed training, independently seeded data order and actual matched hashes, batch throughput/memory arithmetic and its cold-start scope, correctness of new dense full-training/selection/formal-evaluation orchestration, same checkpoint across datasets and epoch0/positive-best semantics. Verify input seals before/after; independently recompute any supported counts/ratios/hash comparisons. Identify real blockers precisely and distinguish source bugs from unsupported high-level claims, runtime-not-yet-exercised orchestration, limited hardware/weight provenance and nonblocking uncertainty. Do not interpret snapshots as current live state and do not claim formal results absent completed files. Need A-F PASS/WARN/FAIL with exact file:line evidence, overall verdict and blocking_findings_count, concrete obligations/claim limits, counts of actual checks, retained failed checker attempts and exact output hash manifest. Record requested reviewer gpt-6-astra/max, actual model/routing/effort UNATTESTED unless independently available; review_independence same-family and acceptance_status provisional. Keep .aris traces/private inputs private. Return your substantive verdict and output paths. Review proceeds while existing training continues, so do not introduce a new GPU gate or restart anything.'''
meta=dict(skill='experiment-audit',run_id='2026-10-15_dense_m0_1435',generated_at=now,agent_id=report['agent_id'],requested_model='gpt-6-astra',requested_reasoning_effort='max',actual_model='UNATTESTED',actual_routing='UNATTESTED',actual_reasoning_effort='UNATTESTED',review_independence='same-family',acceptance_status='provisional',status='complete',verdict='WARN',blocking_findings_count=0,full_host_transcript_available=False)
(trace/'run.meta.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
(trace/'001-dense-m0.request.json').write_text(json.dumps(dict(call_number=1,purpose='fresh-dense-m0-integrity-audit',prompt=prompt,files_referenced=[f['path'] for f in M['files']],requested_model='gpt-6-astra',requested_reasoning_effort='max'),indent=2)+'\n',encoding='utf-8')
(trace/'001-dense-m0.response.md').write_text(md,encoding='utf-8')
(trace/'001-dense-m0.meta.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
events=OUT/'.aris/meta'; events.mkdir(parents=True,exist_ok=True)
(events/'events.jsonl').write_text(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='fresh-dense-m0-integrity-audit',agent_id=report['agent_id'],trace_path=str(trace.relative_to(OUT)),status='ok',private=True))+'\n',encoding='utf-8')
early=[dict(chunk_id='a5e26d',tool_exit_code=1,action='audit_io read current handoff',error='UnicodeEncodeError: gbk cannot encode U+2212 at line5246',raw_streams_available=False,reproduction='runs/read_gbk_reproduction_01'),dict(chunk_id='65698d',tool_exit_code=1,action='run_check forwarding seal --phase after',error='argparse unrecognized arguments: --phase after',raw_streams_available=False,reproduction='runs/runner_argparse_reproduction_01')]
(OUT/'early_tool_failures.json').write_text(json.dumps(early,indent=2)+'\n',encoding='utf-8')
(OUT/'REPRODUCE.md').write_text('''Run only the audit checkers, never the sealed project runners. All paths in this directory are local/private.

`E:/python.exe -B check_artifacts.py`
`E:/python.exe -B check_source_stdlib.py`
`E:/python.exe -B check_orchestration.py`
`E:/python.exe -B check_lineage_and_coverage.py`
`E:/python.exe -B audit_io.py seal --phase after`

For exact stdout/stderr/exit capture, use `run_check.py <new_unique_tag> <checker.py> [arguments]` from this directory. This writes only inside this audit directory. Use a new tag because old attempts are immutable. The NumPy checker is retained as a failed attempt, not required to reproduce the successful stdlib audit. No Torch/model imports or installs are needed.

Input paths pass through a Windows junction to `D:/Program Files/UserCache/gb/codex_tmp`; this explains canonical paths in subprocess receipts. It is the same authorized output directory.

The private JSON leaf inventory contains exact source receipt fields; `.aris` contains the delegated request and full substantive review, not an invented complete host transcript. Do not publish either by default.
''',encoding='utf-8')
print(json.dumps(dict(verdict='WARN',blocking_findings_count=0,counted_assertions=count,reports=['EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json'],trace=str(trace.relative_to(OUT))),indent=2))
