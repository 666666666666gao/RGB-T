# Experiment audit: R1 main versus pure CAD on RGBT210

**Overall verdict: WARN. Integrity status: warn. Blocking findings: 0.**

The supplied accuracy results are reproducible. All **4154 named deterministic checks pass**, and all **1156 sealed inputs remain unchanged**. Both roles cover **210 sequences / 104,706 original frame pairs**, with 104,496 timed post-initialization frames each. Every supplied overall/per-sequence JSON/CSV score, mean curve, attribute aggregate and latency aggregate matches the independent recomputation exactly. There is no self-normalization, missing-sequence substitution or phantom score found in this pair.

Date: 2026-10-15. Fresh delegated reviewer: `/root/r1_pure210_pair_result_integrity_audit`. Requested: `gpt-6-astra`, reasoning `max`, `fork_turns=none`. **Actual model/routing/reasoning/fork mode: UNATTESTED.** Review independence is **same-family**; acceptance status is **provisional**, never cross-family accepted. Deterministic arithmetic is accepted only for the supplied bytes and protocol.

Input root (read-only): `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230`. Output root: `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_pure210_pair_formal_audit_20261015`. Python is 3.10.19 and NumPy 2.2.6; exact interpreter and NumPy module path are in `runtime.json`. No neural import/execution, SSH/network, GPU query or package/environment change was performed.

| Role | Epoch | PR (%) | SR (%) |
|---|---:|---:|---:|
| main | 10 | 87.410354930461 | 63.667766126564 |
| pure | 14 | 88.722375342089 | 64.515171487517 |

The paired 5,000 sequence bootstrap uses seed 20261015, `numpy.random.default_rng` / PCG64, the saved toolkit sequence order, the same 210 resampled named-sequence indices for PR and SR, and linear 2.5/97.5% quantiles. Units below are percentage points, main minus pure.

| Metric | Paired point difference | Marginal 95% percentile interval |
|---|---:|---:|
| PR | -1.312020411628 | [-2.956321814032, 0.212596459914] |
| SR | -0.847405360954 | [-2.002997684355, 0.230742480178] |

Both point estimates favor pure CAD, and both intervals include zero. This is neither evidence of superiority nor evidence of equivalence. It also does not establish significant inferiority at the displayed two-sided 95% level. These are fixed-checkpoint sequence intervals: they exclude training-seed variation and selection uncertainty. Frames within a video are dependent; related events across videos may remain dependent. The two marginal 95% intervals are not a simultaneous confidence region.

The protocol reads GT and predictions as float32, rounds prediction coordinates with `round(0)`, then replaces each first prediction with first GT. Despite the toolkit comment “ignore the first frame,” that frame remains in both numerator and denominator. Strict centers use `(w−1)/2+x`, and IoU uses inclusive pixel edges. PR counts error≤threshold at 0…50 and reports threshold 20; SR counts IoU>threshold at 21 points 0…1 and takes their arithmetic mean. No trapezoidal integration or model-dependent rescaling is introduced. All original annotations stay in the denominator, including the three nonpositive-width/height rows in car10, child4 and man24. The supplied scalar PR/SR definitions and the independent vector implementation agree at all 30,240 sequence-curve values across both roles. The12 attributes are independently reconstructed from the supplied 234-entry flag/name files and intersected with the 210 benchmark names.

## A gt provenance — PASS

Both roles are scored against the same 210 supplied dataset init.txt annotations. The scoring loader replaces the toolkit default GT with the dataset root. Main inference reads only the first annotation; pure inference reads labels for length checking and supplies only labels[0] to initialization. Later frame tracking takes images and prior state, not later GT. All 104706 rows are read and preserved; three rows with nonpositive width/height are kept under the native protocol.

Evidence:

- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/project/cadtrack_abc/score.py:36` — Dataset annotations replace evaluator.seqs_gt; RGBT210 uses init.txt for both streams.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/dataset/basedataset.py:9` — GT files are loaded independently of prediction files.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/project/cadtrack_abc/infer.py:56` — First annotation initializes; step at line65 receives the current image pair and state.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/project/cadtrack_abc/infer_baseline.py:35` — All labels are read for frame-count agreement, but only labels[0] is passed to initialize.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/project/cadtrack_abc/search_actions.py:63` — Runtime search and advance_candidate use causal state and current images.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/documentation/plan_snapshot.json:1379` — Collection declaration identifies the archive and GT inventory; archive bytes/publisher attestation are absent here.
## B score normalization — PASS

No denominator is derived from model-output maxima, minima, means or confidence scores. GT/predictions are loaded as float32; predictions are rounded to integer coordinates, with ties-to-even, then the first predicted row is replaced by first GT. Strict center locations use (w-1)/2+x and integer-inclusive IoU uses x+w-1. PR uses <= at thresholds 0..50 and reports20; SR uses > at 21 linspace thresholds 0..1 and reports the arithmetic mean, not a trapezoidal integral. Every original frame remains in the per-sequence denominator, including initialization and invalid annotations. Sequences are equally weighted. Independent vector arithmetic and the isolated supplied scalar PR/SR source produce identical full per-sequence curves.

Evidence:

- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/utils.py:31` — Default annotation/result dtype is np.float32.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/dataset/basedataset.py:22` — Predictions round(0) before ltwh conversion.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/utils.py:122` — CLE strict pixel-center definition.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/utils.py:162` — Integer-inclusive overlap and ordinary union denominator.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/metrics/metrics.py:177` — PR initialization replacement, <=, full len(res), sequence mean.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/metrics/metrics.py:212` — SR initialization replacement, >, full len(res), mean over sequences/thresholds.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/project/cadtrack_abc/score.py:54` — RGBT210 invokes only PR/SR, scales fractions by100, derives sequence and attribute metrics.
## C result existence — WARN

All supplied raw files, 420 complete prediction series, 420 latency arrays, four shard receipts, both JSON/CSV reports, mean curves and 12 attribute aggregates reproduce exactly. Main best.pth SHA and epoch 10 literal match state/selection/report; the pure checkpoint has collection-time SHA metadata but no bytes. Recorded child exits and completion states agree. Stale baseline_source/timing prose is present in both raw reports. Historical imports, loaded parameter tensors and original controller OS exit are not independently attested by this static bundle. Development scores and20-epoch training are receipt-consistent, not independently reconstructed from missing training/development raw artifacts.

Evidence:

- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/results/main/rgbt210/full_report.json:2` — Completed main epoch 10 report and original denominators.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/results/pure_rgbt210/full_report.json:2` — Completed pure epoch 14 report and original denominators.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/main_state.json:8` — Recorded main checkpoint and configuration hashes checked against supplied bytes.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/main_selection.json:2` — Selected epoch 10 is the maximum of six declared DEV candidate means; best/trained_best coincide.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/main_fit_complete.json:7` — 20 epochs/ 1,600 updates and5092 queries; training seed 42 is recorded at line61.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/pure_checkpoint_identity.json:2` — External checkpoint path/size/hash; explicitly collection-time-only, bytes not collected.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/pure210_state.json:51` — Both inference child exits 0 and score exit0.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/documentation/run_pure_rgbt210_gpu2_remote.stdout.txt:2` — Final original stdout state agrees exactly with pure210_state.json; stderr empty.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/results/pure_rgbt210/full_report.json:239` — Stale two-GPU timing wording.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/results/pure_rgbt210/full_report.json:241` — Stale no-baseline-predictions/no-paired-CI wording.
## D dead code — PASS

The claimed PR/SR functions are on the actual supplied scoring call path, have corresponding complete output fields, and were independently executed offline as isolated numeric definitions. MPR/MSR, GTOT and LasHeR helpers in metrics.py apply to other protocols and are not used for this RGBT210 claim. Training data.iou, future-action utility and synthetic sanity helpers are not substituted for the reported PR/SR. This confirms call-path consistency, not historical import attestation.

Evidence:

- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/project/cadtrack_abc/score.py:53` — Registers raw prediction directory, invokes dataset-specific metrics, writes results at line97.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/dataset/rgbt210_dataset.py:86` — PR dispatches to PR_fun.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/dataset/rgbt210_dataset.py:113` — SR dispatches to SR_fun.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/documentation/run_pure_rgbt210_gpu2_remote.py:57` — Controller invokes cadtrack_abc.score and records successful completion.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/main_state.json:417` — Recorded RGBT210 scoring command and return code.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/project/cadtrack_abc/future_pipeline.py:263` — Generic pipeline inference/scoring call chain; this source is not proof that the generic owner was the particular historical controller.
## E scope — WARN

Evidence establishes the fixed selected main checkpoint and the recorded pure baseline on one common RGBT210 benchmark, with210 equally weighted sequences and104706 frame pairs per role. Main has lower point estimates and both marginal paired 5,000 percentile intervals include zero. Neither superiority, significant inferiority, equivalence, three independently effective modules, matched speedup, training-seed stability nor research-goal completion follows. Main seed 42 is recorded; the pure pretrained training seed is not supplied. Repeated98-DEV selection is conditioned on rather than included in uncertainty. No external control-job status is inferred from the static bundle.

Evidence:

- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/main_fit_complete.json:58` — Recorded main training hyperparameters, seed 42.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/main_selection.json:56` — Repeated98 development selection explicitly acknowledged.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/documentation/HANDOFF_snapshot.md:5` — Current overview says the research targets are unmet.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/documentation/HANDOFF_snapshot.md:7` — Current overview already rejects matched baseline speed and independent module-effect claims.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/documentation/plan_snapshot.json:385` — R1/R2/R3 and independent-contribution requirements are separate stages.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/main_state.json:316` — Historical RGBT210 inference uses two workers on physical GPU2.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/pure210_state.json:7` — Pure run likewise records physical GPU2 and two workers, in a different time window.
## F evaluation type — PASS

Both evaluations compare predictions with supplied dataset-provided annotation files. These are real_gt evaluations within the supplied-byte provenance limit, not synthetic agreement/proxy tests. Initial-frame GT use is the benchmark initialization convention, and subsequent labels are used for offline scoring.

Evidence:

- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/project/cadtrack_abc/score.py:38` — RGBT210 ground-truth annotation paths.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/toolkit/rgbt/dataset/basedataset.py:12` — Dataset annotation file reads.
- `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/pure210_pair_audit_inputs_1230/raw_input_manifest.json:4` — Supplied annotation source paths, lengths and hashes; all 1,147 manifest entries verified.

## Checkpoint and completion evidence

Main `selected_checkpoint/best.pth` is 2,603,720 bytes, SHA256 `a14130fafbc7eb49a17a3834fe8a43855f3ed0833ba576e737e74d1579965ad8`. Its unique pickle epoch literal is 10, matching selection, both receipts, report and controller state. Inspection used zip/pickletools without unpickling tensors or importing torch. The hash also matches the main controller declaration. All 14 recorded source hashes and the configuration hash match the supplied files.

Pure checkpoint `/home/gaob/CADTrack_ep0014.pth.tar` has recorded collection-time size 1,236,957,664 bytes and SHA256 `cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731`. The bytes are absent; epoch 14 and 250 loaded tensors are receipt declarations supported by the source loader, not independently loaded-state attestation. Main also depends on that external base checkpoint.

The four inference shard receipts each contain 105 disjoint named sequences; shard frame totals are 49,860 and 54,846 for each role. Every per-sequence count and latency sum agrees. Recorded main/pure inference exits are 0/0 and score exits 0. Pure original stdout ends in exactly the supplied complete state; original stderr is empty. The observer recorded controller absence at 04:25 UTC. These static facts do not independently attest an OS-level controller exit code or current external jobs. Training 20 epochs / 1,600 updates and six-candidate DEV selection are internally consistent declarations; missing raw training and DEV traces are not reconstructed here. Main fit receipt `formal_metrics_complete=false` is a stage-local training receipt, superseded for this role by later completed selection/state/result artifacts.

## Timing

| Role | Recorded FPS | Mean latency (ms) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|
| main | 21.495285474 | 46.521829226 | 46.474418996 | 50.684119502 |
| pure | 28.327086489 | 35.301900899 | 34.704470512 | 42.590589008 |

FPS is 104,496 divided by the sum of recorded worker-frame seconds. It is not concurrent wall-clock throughput. Timers include decode/crop/visual forwards/decision/state or template update and ending CUDA synchronization; initialization and prediction serialization are excluded. Both supplied controller records describe two inference workers on physical GPU2, at different times. The raw reports’ “two-GPU” wording is stale. These measurements do not support matched speedup or lower complete-system cost.

## Nonblocking findings and actions

**N1 (stale_report_metadata)**. Both reports line239 say concurrent two-GPU run, but both controller records declare two inference workers on GPU2. Both line241 say no baseline predictions rerun and no paired CI; this is stale template prose for the newly completed pure210 role and does not describe this new paired audit. Action: Keep raw reports immutable. In derived summaries identify the roles, recorded two-workers-on-GPU2 execution, timing scope, and this paired-bootstrap artifact explicitly.

**N2 (historical_runtime_attestation)**. Main checkpoint bytes and source/config snapshots are verifiable, but no independent historical loaded-state/import trace is supplied. Pure checkpoint bytes are absent and its SHA is a collection-time receipt. Strict-load flags and 250 loaded tensors are declarations supported by source, not a new neural replay. Action: Qualify historical runtime identity as UNATTESTED; retain the supplied hash/receipt evidence without claiming more.

**N3 (publisher_authentication)**. The dataset collection manifest and210 GT files agree with the seal. Original archive bytes, image files, independent publisher hashes/signatures and the installed toolkit default GT distribution are not in the bundle. Archive URL/hash and historical agreement with installed GT are recorded declarations. Action: Say supplied RGBT210 dataset annotations and supplied toolkit protocol; do not imply independent authentication of the original publisher distribution.

**N4 (scope_and_selection)**. One fixed main checkpoint and one pretrained pure role do not establish seed stability or any independent module contribution. Full98 DEV histories, raw training logs, main LasHeR/RGBT234 raw evaluation bundles, and external control-job evidence are outside this pair audit. Action: Restrict numerical claims to this fixed RGBT210 pair; keep the research goal unmet and external control status unassessed.

**N5 (bootstrap_dependence)**. Sequence bootstrap conditions on selected checkpoints and treats named sequences as exchangeable. Frames within videos and potentially related events across videos are dependent. Marginal 95% percentile intervals do not represent training-seed variation or simultaneous 95% joint coverage. Action: Publish the order/seed/replicates and dependence/selection qualifications with every interval.

**N6 (review_attribution)**. Requested gpt-6-astra/max/fork-none is recorded; this reviewer has no independent host attestation of actual model, reasoning effort or fork routing. Action: Keep actual model/routing/effort UNATTESTED and review_independence same-family, acceptance_status provisional.

The nonblocking classification permits qualified reporting of these already supplied numerical results. It does not waive the evidence needed for broader effectiveness or research-goal claims.

## Claim impact

- **C1 supported** — The two completed supplied RGBT210 reports are numerically correct. Under the exact supplied protocol, with the original210 sequences and104706 frame denominators.
- **C2 supported_with_qualifiers** — Main-minus-pure fixed-checkpoint PR/SR differences and paired sequence intervals. Paired5000 seed 20261015, toolkit sequence order, PCG64 and linear marginal percentiles; see paired_bootstrap_5000.json.
- **C3 unsupported** — Main improves RGBT210 accuracy over pure CAD. Both point differences are negative; both 95% intervals include zero.
- **C4 unsupported** — The main and pure models are equivalent, or main is significantly inferior. Intervals crossing zero establish neither equivalence nor a significant difference.
- **C5 unsupported** — All three modules independently contribute gains. No full factorial or matched individual-module controls in this pair.
- **C6 unsupported** — Main provides matched speedup or lower system cost. Recorded worker latency aggregates are not a matched benchmark or wall-clock concurrent throughput.
- **C7 unsupported** — Training-seed stability or robust repeated-training gains. Sequence resampling does not resample training runs.
- **C8 unattested** — Original publisher data or exact historically loaded parameters/imports authenticated. 
- **C9 not_assessed** — Current external control-job state. Static snapshots are not live status.
- **C10 unsupported** — The overall research goal is achieved. 
- **C11 supported_with_qualifiers** — Main best.pth bytes match the recorded selected epoch 10 identity. SHA256 and unique pickle epoch literal verified without torch import/unpickling; not historical load attestation.

## Reproduction and forensic record

`verify_pair.py` independently reads the complete manifest/protocol, all 210 GT files, both sets of210 prediction files and latency arrays, all four receipts and both reports. It then cross-checks numeric-only AST extracts of the supplied PR/SR source. There was one numerical verification execution, with actual return code 0 and no failed numerical attempts. No mismatch was erased, no tolerance was changed, and no input was modified. The full manifest has 1,147 entries; nine supplemental files are covered by the 1,156-file initial seal. Seventy supplied Python files parse successfully. Direct semantic source inspection concentrates on the requested scoring, inference, selection, causal runtime and relevant vendor/config dependencies; it does not claim a line-by-line review of every historical handoff entry.

Artifacts: `deterministic_verification.json`; `recomputed_results.json`; `main_*` and `pure_*` per-sequence, curve, attribute and latency CSVs; `aggregate_latency_statistics.json`; `paired_bootstrap_5000.json`; `paired_bootstrap_5000_replicates.csv`; `paired_sequence_differences.csv`; `ground_truth_inventory.csv`; `checkpoint_identity_verification.json`; `input_hashes_before.json` / `input_hashes_after.json`; `execution/001_verify_pair.*`; `source_read_tool_trace*.json`; the request/response trace under `.aris/traces/experiment-audit/2026-10-15_run01/`. Final validation and output SHA manifest are separate artifacts produced after report writing.

Large exploratory documentation outputs were display-truncated and this limitation is retained in the tool trace. All numerical inputs and the entire raw manifest/protocol were nevertheless read and checked in full by the verifier. Early exploratory tool outputs remain partly in the host transcript; this audit cannot export a complete host transcript or attest host routing. No raw GT/prediction coordinates, checkpoint contents or per-frame latency arrays are emitted into the audit reports.
