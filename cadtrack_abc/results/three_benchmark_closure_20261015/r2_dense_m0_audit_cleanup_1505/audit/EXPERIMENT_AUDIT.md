# Experiment Audit — dense R2 M0 and launch snapshot

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
