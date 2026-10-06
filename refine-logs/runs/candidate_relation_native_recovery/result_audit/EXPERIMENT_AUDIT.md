# Experiment audit — selected native ABC result

Date: 2026-10-06T13:18:16.987Z. Overall: **WARN** (advisory; actual results may be recorded). Configured reviewer: **gpt-6-astra / max**, fresh native Codex agent; runtime backend/model **unattested**; **same-family / provisional**. No NN, score or bootstrap rerun; no experiment-source edits.

The selected reconstructed epoch5 genuinely has saved complete reports for LasHeR **245/220703** and RGBT234 **234/116649**. All eight full shard receipts/configs, native means/curves and per-sequence tables agree; the published derivative files agree in all **5 metric, 81 attribute and 20 paired-comparison rows**. Five +2 pp acceptance remains **unmet (0/5)**.

| Metric | Local GOLA % | Selected ABC % | Delta pp | +2 met |
|---|---:|---:|---:|---|
| lasher PR | 76.598449 | 78.301553 | +1.703104 | No |
| lasher NPR | 73.101973 | 74.657486 | +1.555513 | No |
| lasher SR | 61.037425 | 62.339574 | +1.302148 | No |
| rgbt234 MPR | 91.778858 | 92.035427 | +0.256569 | No |
| rgbt234 MSR | 69.232593 | 69.454475 | +0.221882 | No |

## A. Ground truth provenance: PASS

Both native benchmarks use dataset annotations, not model-output references. Independent read-only server inspection verified all native evaluator GT arrays against every dataset annotation: LasHeR 245 sequences/220703 frames, RGBT234 234/116649. Online evaluation reads only the first annotation for initialization. Developer/reconstruction prediction equality is a separately labelled consistency check, not accuracy.

Evidence: `research/evaluate_recoverability.py:214`; `research/evaluate_recoverability.py:221`; `research/collect_core_metrics.py:114`; `research/collect_core_metrics.py:121`; `research/collect_candidate_metrics.py:28`; `evaluation.py:46`; `evaluation.py:81`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/reconstruction_verification.json:24`.

## B. Score normalization: PASS

No division of native accuracy by the model-own maximum, mean, or range. Native functions produce per-sequence threshold fractions; collector reports percent, equal-sequence means and plain paired differences. RGBT234 maximum means choosing the better RGB/TIR annotation agreement, not a prediction normalization denominator. Motion NLL/ADE diagnostics use the forecast reference coordinate system and predicted distribution scale; these are explicitly separate from native tracking accuracy.

Evidence: `research/collect_core_metrics.py:87`; `research/collect_core_metrics.py:171`; `research/paired_sequence_bootstrap.py:45`; `research/collect_recoverability_metrics.py:254`; `research/collect_abc_metrics.py:57`; `/data/gb/envs/gola/lib/python3.10/site-packages/rgbt/metrics/metrics.py:36`; `/data/gb/envs/gola/lib/python3.10/site-packages/rgbt/metrics/metrics.py:71`.

## C. Existing results, values and completion: WARN

Actual full native completion and all five negative-to-target results are supported. Saved-artifact consistency verification passed: all eight shard configs/receipts, disjoint complete sequence union, curve and CSV means, eight pair reports/5000-resample CI records, and derivative 5/81/20 rows. Pre-run selected_model.native_metrics_completed=false remains inside terminal progress; this is a historical selection snapshot, not evidence of missing native results. Generic core mechanism_measurements_pending strings also remain although the separate mechanism reports are complete. Preserve originals and use native/complete_metrics.json plus the dated completion summary as current authority.

Evidence: `refine-logs/runs/candidate_relation_native_recovery/complete/training/native/complete_metrics.json:2`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/native/complete_metrics.json:14`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/selected_model.json:10`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/progress.json:70`; `research/collect_core_metrics.py:140`; `refine-logs/EXPERIMENT_TRACKER.md:152`; `docs/HANDOFF_20261002.md:3`; `refine-logs/runs/candidate_relation_native_recovery/result_audit/.aris/traces/experiment-audit/20261006_selected_native/saved_artifact_verification.json:1`.

## D. Metric execution versus dead code: PASS

Core native metrics, full attributes/curves, paired bootstrap, plot exports, and GT mechanism diagnostics are called by the native controller and have actual nonempty output artifacts. Trainer evaluation is called before training, each epoch, and during strict reload; original 61 records and reconstructed six-record prefix match. The optional state_commit_geometry function is not active for this selected ABC configuration (state_commit_model=null and no state counters); no current state-commit result is claimed from that branch.

Evidence: `scripts/run_selective_state_native.py:150`; `scripts/run_selective_state_native.py:156`; `scripts/run_selective_state_native.py:158`; `research/collect_core_metrics.py:171`; `research/train_recoverability.py:228`; `research/train_recoverability.py:284`; `research/collect_recoverability_metrics.py:91`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/selected_model.json:7`.

## E. Scope and contribution: WARN

Full native datasets, not pilots: one developer-selected reconstructed epoch5, seed42, used on both datasets; 19+12 attributes. Original chosen run completed 60 epochs/480 updates, reconstruction only the selected prefix5/40. One of eight models was chosen on repeatedly used developer98; not untouched confirmation. Bootstrap is fixed-checkpoint sequence uncertainty, not training-seed stability or a joint five-metric guarantee. Relation/ABC gains cannot be attributed from baseline gains: versus gross parent the LasHeR deltas are +0.285459 PR/+0.273702 NPR/+0.266914 SR while RGBT234 changes are -0.830934 MPR/-0.567907 MSR; every corresponding CI includes zero. All five baseline+2 point targets fail. A single best checkpoint remains the user acceptance rule; no three-seed condition is added.

Evidence: `scripts/run_candidate_relation_training.py:166`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/selected_model.json:26`; `refine-logs/runs/candidate_relation/complete_v2/training/fit_full/relations_lr5/completion.json:3`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/reconstructed_epoch5/completion.json:3`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/reconstruction_verification.json:24`; `research/paired_sequence_bootstrap.py:24`; `scripts/run_selective_state_native.py:167`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/native/lasher/gross_parent_paired_report/paired_bootstrap.json:21`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/native/rgbt234/gross_parent_paired_report/paired_bootstrap.json:21`; `docs/HANDOFF_20261002.md:19`.

## F. Evaluation classification: PASS

Native tracking scores and full-video localization diagnostics are real_gt. Candidate recall, wrong-template writes and memory contamination are GT-localization proxies, not semantic identity labels. Cached short-horizon utility is a real-GT-labelled policy-contingent diagnostic, not native accuracy. Reconstruction parity is output-derived consistency only (synthetic_proxy in the limited output-reference taxonomy), explicitly not a performance evaluation. No human-eval, simulation-only benchmark or self-supervised accuracy claim is supported or asserted.

Evidence: `research/collect_recoverability_metrics.py:178`; `research/collect_recoverability_metrics.py:180`; `research/collect_recoverability_metrics.py:182`; `research/recoverability_modules.py:226`; `research/train_recoverability.py:226`; `refine-logs/runs/candidate_relation_native_recovery/complete/training/reconstruction_verification.json:24`.

## Protocol qualifications

The repository wrapper calls the installed rgbt native functions, not a new approximate scoring formula. Native result loading rounds XYWH to integers (server '/data/gb/envs/gola/lib/python3.10/site-packages/rgbt/dataset/basedataset.py:22'). The LasHeR package replaces the first result with GT and retains its built-in handling of invalid annotations: NPR/PR set errors to -1 for any nonpositive GT component, whereas SR sets overlap to -1 (server 'rgbt/metrics/metrics.py:253', ':266', ':309', ':351'). The collector preserves NPR→PR→SR call order ('research/collect_core_metrics.py:19', ':171'). Raw-box, valid-GT, initialization-excluded mechanism IoU is therefore a distinct diagnostic and must not replace native scores. Independent dataset-GT equality verified 245 LasHeR and both annotations of 234 RGBT234 sequences; no native scoring function was called during this audit.

## Claim impact and action

Actual native performance and complete reporting are supported. Baseline gains do not establish a new candidate-relation or individual A/B/C contribution: all gross-parent CIs include zero, with RGBT234 lower than the parent. No stable-seed, untouched-confirmation, semantic-identity recovery, persistent rejected-trajectory, or arbitrary historical repair claim follows. One best checkpoint is sufficient under the user's rule; this audit adds no seed requirement.

The original selected weight's deletion and historical native skip remain visible in 'refine-logs/runs/candidate_relation/complete_v2/training/native_not_replayed.json:2' and 'refine-logs/runs/candidate_relation/complete_v2/training/unused_own_weight_cleanup.json:52'; they are not the current native status. The reconstruction receipt explicitly limits parity to saved queries and developer98 ('refine-logs/runs/candidate_relation_native_recovery/complete/training/reconstruction_verification.json:24'). The new dated handoff and tracker correctly mark native execution complete while the goal stays unmet; older launch paragraphs are historical. Treat 'native/complete_metrics.json' and the dated derivative completion summary as the current authority, and preserve raw pre-run artifacts. Generic core-report pending notices do not nullify the separately completed mechanism reports.

Read/verification scope: all ten supplied source files read directly; relevant imported GT/metric/utility helpers inspected; all supplied saved JSON data parsed; selected original/rebuilt configuration and epoch0–5 histories matched; all eight full shard configs/completions checked; eight native pair reports/CI files and actual plot/CSV outputs checked; terminal result and newly published handoff/summary checked. No independent re-execution of historical tensor/trajectory parity, training or scoring. The verification script checks saved values only.

Trace: 'refine-logs/runs/candidate_relation_native_recovery/result_audit/.aris/traces/experiment-audit/20261006_selected_native/'. Deterministic evidence: 'saved_artifact_verification.json', 'remote_gt_verification.txt'.
