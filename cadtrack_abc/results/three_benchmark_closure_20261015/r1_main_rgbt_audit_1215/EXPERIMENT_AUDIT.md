# Experiment integrity audit — R1 main RGBT formal results

**Overall verdict: WARN. 0 blocking issues for the qualified arithmetic report.**

Requested reviewer: `gpt-6-astra`, reasoning `max`. Actual routing/model/effort: **UNATTESTED**. Review independence: **same-family**; acceptance: **provisional**. Deterministic numerical checks passed; this does not certify the research objective.

Input root: `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_rgbt_audit_inputs_1150`. All source references below are relative to that root. Generated: 2026-10-15T04:15:30.165076+00:00.

| Prediction set | Sequences / prediction rows | Precision, % | Success, % |
|---|---:|---:|---:|
| Main selected epoch10, RGBT210 | 210 / 104706 | PR 87.41035493046077 | SR 63.667766126563585 |
| Main selected epoch10, RGBT234 | 234 / 116649 | MPR 89.15954701779948 | MSR 65.93388612904715 |
| Supplied pure CAD epoch14, RGBT234 | 234 / 116649 | MPR 89.88130707509919 | MSR 66.31030440376779 |

All six overall numbers, 1356 per-sequence scalar scores, 48816 per-sequence curve cells, 216 mean-curve cells, 72 attribute scores, original CSV values and latency summaries match exactly with the actual existing NumPy2.2.6 runtime. The audit executed the unmodified toolkit utility source and exact scoring-class AST bodies, then checked independently derived vectorized geometry. It did not import project neural modules or deserialize weights.

## A–F findings

### A. Ground-truth provenance: WARN

All678 supplied annotation files were read. RGBT210 uses210 init.txt files/104706 frames. RGBT234 uses234 visible.txt and234 infrared.txt files/116649 paired frames. Supplied GT contains no model-output reference substitution in the scoring path.

Inference source consumes the first annotation for initialization and subsequent images/state for model.step; after initialization, main infer.py never reads later annotation rows. Pure inference loads labels but uses labels[0] only. Source inspection is not runtime neural attestation.

RGBT234 was sourced from a mirror archive according to its download source, and RGBT210 from a downloaded archive according to extraction source/plan. Original archive bytes, publisher signatures, source images and an independently authenticated upstream annotation copy are absent. Therefore official-distribution identity and visual annotation correctness are not independently certified. Executor manifests establish byte associations, not semantic truth.

Evidence: `project/cadtrack_abc/score.py:35`, `project/cadtrack_abc/score.py:39`, `toolkit/rgbt/dataset/basedataset.py:9`, `project/cadtrack_abc/infer.py:56`, `project/cadtrack_abc/infer.py:65`, `project/cadtrack_abc/infer_baseline.py:35`, `project/cadtrack_abc/infer_baseline.py:45`, `project/cadtrack_abc/download_rgbt234.py:16`, `project/cadtrack_abc/download_rgbt234.py:45`, `project/cadtrack_abc/results/rgbt210_resume_20261015/completed_transfer_0646/extract_verify_official_rgbt210_remote.py:44`.

### B. Normalization and exact installed protocol: PASS

No prediction-statistic normalization. Per-sequence curves divide counts by that sequence frame count; the benchmark averages sequences equally. Precision uses CLE<=t for51 thresholds0..50 and reports index20. Success uses IoU>t for21 thresholds0..1 and reports the arithmetic mean of all21 values, not trapezoidal AUC. Percentages multiply by100.

Raw predictions are written at3 decimals, loaded as float32, then rounded to integer via NumPy round(0). Source uses ltwh, center x+(w-1)/2, inclusive endpoint IoU with area w*h. Main210 PR/SR explicitly replace the first predicted box by GT yet retain the frame in the denominator. RGBT234 MPR/MSR do not overwrite it; all supplied first predictions already equal visible GT.

RGBT210 loads the same init.txt for both modality entries but PR/SR use visible GT only. RGBT234 uses min(CLE_visible,CLE_infrared) and max(IoU_visible,IoU_infrared) independently per frame;87479/116649 paired GT rows differ across modalities. These are distinct dataset definitions, not interchangeable PR/SR and MPR/MSR.

All raw/rounded predicted widths/heights are positive and all predictions/latencies finite. RGBT210 has3 nonpositive-width/height GT rows, retained unchanged in the native denominator; no LasHeR validity filter was transplanted. Nonpositive x/y alone is not filtered for either RGBT dataset.

Actual NumPy2.2.6 reproduces every reported curve and scalar exactly. NumPy1.26.4 legacy scalar promotion shifts strict-success boundary counts: +0.01186474091734624pp SR for210,+0.013693544576128147pp MSR for main234,+0.016702509770652796pp MSR for pure234 when moving from1.26.4 to2.2.6. Both original runs and all changed cells are preserved. No thresholds, GT, source metric or denominators were changed. Historical remote package identity remains documented, not independently attested.

Evidence: `toolkit/rgbt/utils.py:31`, `toolkit/rgbt/dataset/basedataset.py:22`, `toolkit/rgbt/dataset/rgbt210_dataset.py:18`, `toolkit/rgbt/dataset/rgbt210_dataset.py:21`, `toolkit/rgbt/dataset/rgbt234_dataset.py:18`, `toolkit/rgbt/dataset/rgbt234_dataset.py:22`, `toolkit/rgbt/metrics/metrics.py:28`, `toolkit/rgbt/metrics/metrics.py:62`, `toolkit/rgbt/metrics/metrics.py:177`, `toolkit/rgbt/metrics/metrics.py:212`, `toolkit/rgbt/utils.py:122`, `toolkit/rgbt/utils.py:162`, `documentation/HANDOFF_snapshot.md:4945`.

### C. Files, numbers, completion and lineage: WARN

678 prediction text files and678 latency arrays exist: main210/234 and pure234. All678 sequence-run entries,338004 prediction rows,337326 initialization-excluded timings, six shard receipts,1356 per-sequence metric values,48816 per-sequence curve cells,216 mean-curve cells and72 attribute metric values were recomputed. JSON and original CSV values agree exactly. Raw attribute text independently reconstructs all36 reported attribute groups; exported membership was not merely trusted.

Selected best.pth is2603720 bytes, SHA256 a14130fafbc7eb49a17a3834fe8a43855f3ed0833ba576e737e74d1579965ad8; it matches main controller and final observation. Config and all14 launch-listed source hashes match. Selected epoch10 is the maximum among all six reported DEV sequence-IoU summaries and agrees with all main receipts/reports; trained_best equals best.

Selection metadata itself has no independent weight digest. Candidate epoch10 bytes and raw98-video DEV predictions/annotations are absent here, so candidate-to-best byte copying and DEV score recomputation are not re-attested. The external pure/base pretrain bytes are absent. No weights were deserialized or network loaded.

Main controller records six inference children and three scoring children exit0 across three datasets. For the two newly audited RGBT datasets, all four inference and two score child codes are0, exact frames/files exist, and the final observation reports controller PID absent. There is no parent controller OS exit-code receipt in this bundle; absence of a PID alone does not prove successful exit. Pure234 has both complete shard receipts and full outputs but no supplied pipeline OS-exit record.

Source-at-launch coverage omits actual runtime dependencies search_actions.py,counterfactual.py,ported_modules.py,vendor files and the installed scoring toolkit/runtime. Their collected bytes are hashed but cannot retrospectively prove the historical import graph or the loaded tensor identity.

Evidence: `results/main/rgbt210/full_report.json:2`, `results/main/rgbt234/full_report.json:2`, `results/pure_rgbt234/full_report.json:2`, `main_benchmark_state.json:8`, `main_benchmark_state.json:13`, `main_benchmark_state.json:264`, `main_benchmark_state.json:407`, `main_benchmark_state.json:454`, `main_observation_1150.json:4`, `selection_after_formal.json:2`, `selection_after_formal.json:54`, `project/cadtrack_abc/select_future_actions.py:39`, `project/cadtrack_abc/infer.py:36`, `project/cadtrack_abc/baseline.py:27`, `project/cadtrack_abc/results/r1_selected_and_formal_20261015/0700/run_r1_main_three_benchmarks_remote.py:26`.

### D. Called metrics versus unused paths: PASS

The current controller invokes cadtrack_abc.score after inference. Its measured mapping calls PR/SR for210 and MPR/MSR for234. Dataset methods dispatch to the exact metric classes; all corresponding outputs occur in real JSON/CSV files and match independent recomputation.

NPR/LasHeR metrics and GTOT metrics exist in the shared toolkit but are outside these RGBT calls by design; no missing RGBT NPR claim was manufactured. Development IoU and training utility are separate source paths and are not used as formal benchmark metrics.

score.py only calls mechanism_summary for LasHeR. RGBT per-frame mechanism JSON generated by infer.py is not in these supplied inputs, and there are no RGBT mechanism-report claims supported by this audit. No independent A/B/C contribution can be inferred from live code or a trained-head receipt.

Evidence: `project/cadtrack_abc/results/r1_selected_and_formal_20261015/0700/run_r1_main_three_benchmarks_remote.py:102`, `project/cadtrack_abc/score.py:53`, `project/cadtrack_abc/score.py:54`, `project/cadtrack_abc/score.py:55`, `project/cadtrack_abc/score.py:63`, `project/cadtrack_abc/score.py:70`, `project/cadtrack_abc/score.py:75`, `project/cadtrack_abc/score.py:84`, `project/cadtrack_abc/baseline_pipeline.py:39`, `toolkit/rgbt/dataset/rgbt210_dataset.py:86`, `toolkit/rgbt/dataset/rgbt234_dataset.py:104`.

### E. Scope and claim ceiling: WARN

Supported scope is the fixed selected main search_actions epoch10/seed42 reported RGBT210 and RGBT234 results, plus the supplied pure CAD epoch14 RGBT234 prediction set. Training receipt says20 epochs/1600 updates on search+future-action heads with native ABC/motion coordinates frozen; no full visual/ABC retraining is established by this bundle. One selected seed is expressly allowed; multiple seeds are not an acceptance requirement.

A separate completed LasHeR audit exists and the current main controller now records all three formal benchmarks complete. This supersedes earlier snapshots marking main RGBT work in progress. This audit does not re-audit raw LasHeR predictions; it does not infer whether other control roles or pure210 are currently running/completed from old flags or missing files. Those roles are unverified here.

Main minus pure234 is−0.7217600572997469pp MPR and−0.37641827472065537pp MSR. Both paired-sequence percentile95% bootstrap intervals include0. These are fixed-checkpoint sequence intervals, not training-seed, causal-module or independent-dataset evidence. No LasHeR baseline paired CI was constructed.

The main timing.scope text says concurrent two-GPU run, but controller launch assigns both main shards to physicalGPU2. Recomputed fps is calls/sum(worker-frame-seconds), not concurrent system throughput. Inference timer includes decode/crop/forwards/update and synchronization; excludes initialization, initial sync wait, subsequent diagnostic conversion/serialization, and output writing. Pure timing used different execution conditions. No matched speedup or lower complete-system cost follows.

No evidence establishes independent effectiveness of all three modules, full A+B+C/equal-budget/eight-combination results, publisher-certified protocol equivalence, cross-family acceptance or goal completion. The plan and controller explicitly leave the research goal unmet.

Evidence: `fit_complete.json:2`, `fit_complete.json:6`, `fit_complete.json:61`, `documentation/plan_snapshot.json:351`, `documentation/plan_snapshot.json:376`, `documentation/plan_snapshot.json:387`, `documentation/plan_snapshot.json:389`, `documentation/HANDOFF_snapshot.md:5175`, `documentation/HANDOFF_snapshot.md:5189`, `documentation/HANDOFF_snapshot.md:5195`, `main_benchmark_state.json:493`, `results/main/rgbt234/full_report.json:239`.

### F. Evaluation type: PASS

All three audited evaluations compare model predictions to supplied dataset annotation files; the pure model is a prediction reference for paired differences, never the ground truth. None of these three is synthetic_proxy,self_supervised_proxy,simulation_only or human_eval. Classification is conditional on the supplied dataset provenance limits in A.

Evidence: `project/cadtrack_abc/score.py:39`, `project/cadtrack_abc/score.py:55`, `toolkit/rgbt/metrics/metrics.py:31`, `toolkit/rgbt/metrics/metrics.py:185`.

## Runtime-boundary discrepancy preserved

The first run used Python3.12.6 and NumPy1.26.4 as initially requested. It completed exit1 with seven named check failures: six success-metric/derived arithmetic mismatches plus an overstrict auditor assertion about invalid dataset GT. It did not expose nonfinite predictions or change inputs. The later run used a parent-authorized, already-existing Python3.10.19/NumPy2.2.6 interpreter, matching the documented warm evaluation version, and completed exit0 with all1437 named checks passing. No packages were installed and no environment files were changed.

| Set / success metric | NumPy1.26.4 | NumPy2.2.6 / report | Difference, pp | Changed curve cells / affected sequences |
|---|---:|---:|---:|---:|
| Main210 SR | 63.65590138564624 | 63.667766126563585 | +0.01186474091734624 | 126 / 91 |
| Main234 MSR | 65.92019258447102 | 65.93388612904715 | +0.013693544576128147 | 145 / 101 |
| Pure234 MSR | 66.29360189399713 | 66.31030440376779 | +0.016702509770652796 | 150 / 101 |

The source loads float32. NumPy2 retains float32 in scalar arithmetic and compares the resulting array against float64 linspace thresholds; NumPy1.26 legacy promotes several scalar expressions to float64. For example, float32(0.2) is 0.20000000298023224 and exceeds the float64 threshold0.2; exact-double0.2 does not. Success uses strict `>`. There are323/422/477 frame-threshold success-count differences respectively. Precision curves are unchanged. This numerical dependence is disclosed rather than hidden by a tolerance or protocol edit.

Independent versus toolkit IoU error is0; maximum CLE rounding difference is3.0517578125e-5 pixels because square-root implementations differ, with zero threshold/count/curve differences. Integer GT and all original denominators remain intact. The3 invalid-width/height GT rows are `ground_truth/rgbt210/car10/init.txt:134`, `ground_truth/rgbt210/child4/init.txt:90`, and `ground_truth/rgbt210/man24/init.txt:369`. No raw coordinates were emitted.

Evidence: `runtime_boundary_diagnosis.json`, `numpy_runtime_curve_differences.csv`, `attempt01_numpy126_legacy/`, and the original execution receipts03/05. The prepared NEP50 compatibility script was never executed; the final accepted arithmetic uses actual NumPy2.2.6.

## Paired RGBT234 uncertainty

| Metric | Main minus pure, pp | Paired-sequence percentile95% interval, pp | Wins / ties / losses |
|---|---:|---:|---:|
| MPR | -0.7217600572997469 | [-2.2697288973079597, 0.8176238517557154] | 53 / 127 / 54 |
| MSR | -0.37641827472065537 | [-1.4486524612226475, 0.7057360967414762] | 121 / 5 / 108 |

The exact rule is5000 paired resamples of234 named sequences with replacement, retaining the main and pure pair, using NumPy2.2.6 `default_rng(20261015)` / PCG64 and identical draws for both metrics. Both benchmark metrics are sequence-equal means, so resampling their paired per-sequence scalars reproduces the estimand. Intervals use `quantile([.025,.975], method="linear")`. Related videos can violate the exchangeable-sequence assumption. Both intervals include0; neither establishes improvement or equivalence. These are fixed-checkpoint sequence intervals, not multi-seed training stability. No LasHeR paired-baseline interval was constructed.

## Issues and supported claim ceiling

- **W-TIMING — Main timing metadata misstates GPU topology and throughput scope.** Correct a qualified presentation copy to two main inference workers sharing physicalGPU2; retain original report/SHA and numbers; label fps as calls per summed worker frame seconds. Do not use it as matched speedup. Evidence: `results/main/rgbt210/full_report.json:239`, `results/main/rgbt234/full_report.json:239`, `project/cadtrack_abc/results/r1_selected_and_formal_20261015/0700/run_r1_main_three_benchmarks_remote.py:92`, `project/cadtrack_abc/infer.py:63`, `project/cadtrack_abc/infer.py:68`.
- **W-RUNTIME-PRECISION — Strict-success results depend on NumPy scalar promotion.** Record NumPy2.2.6 with these results. Preserve first1.26.4 mismatch; do not silently replace protocol or inflate tolerances to hide it. Evidence: `documentation/HANDOFF_snapshot.md:4945`, `toolkit/rgbt/utils.py:185`, `toolkit/rgbt/metrics/metrics.py:75`, `toolkit/rgbt/metrics/metrics.py:228`.
- **W-LINEAGE — Weight bytes and subset source hashes do not attest historical neural execution.** Keep historical runtime imports, external base checkpoint identity, candidate-to-best copy and parent exit status explicitly partial/unattested. Evidence: `main_benchmark_state.json:13`, `project/cadtrack_abc/infer.py:36`, `project/cadtrack_abc/baseline.py:27`.
- **W-PROVENANCE — Dataset/toolkit origin authentication is limited to supplied source and copies.** Describe results as reproduced under the supplied installed toolkit/annotations; do not upgrade hashes or executor manifests to original-publisher semantic certification. Evidence: `project/cadtrack_abc/download_rgbt234.py:16`, `project/cadtrack_abc/score.py:39`.
- **W-SCOPE — Arithmetic success does not close the research objective.** Keep three-module effectiveness, matched efficiency and unmet LasHeR+6 targets unproven; describe absent control/pure210 role evidence as unavailable in this bundle rather than declaring actual current status. Evidence: `documentation/plan_snapshot.json:387`, `documentation/plan_snapshot.json:389`, `main_benchmark_state.json:493`.
- **W-REFERENCE-LABEL — Generic main report baseline_source field is LasHeR-specific.** Qualify main report presentation so the supplied pure234 evaluation is identified separately; the old no-baseline/no-paired-CI sentence is only about the user LasHeR reference. Evidence: `results/main/rgbt234/full_report.json:241`, `results/pure_rgbt234/full_report.json:241`.

Blocking integrity issues for the stated arithmetic scope: **none**. Unsupported broader claims remain outside acceptance; WARN is not permission to state that the research goal or three-module efficacy passed.

The new raw audit closes main RGBT210/RGBT234 arithmetic and the supplied pure234 comparison. The separate LasHeR audit remains separate. Current main completion evidence supersedes older in-progress snapshots, while absent other roles are explicitly unverified. Selected weight hashing authenticates the provided bytes against the recorded digest; it does not prove historical loading or inference.

## Artifacts and immutability

- `EXPERIMENT_AUDIT.json` — machine verdict, A–F evidence, limits and claims.
- `deterministic_verification.json` — all1437 named checks and per-run summaries.
- `recomputed_per_sequence.csv`, `recomputed_sequence_curves.csv`, `recomputed_mean_curves.csv`, `recomputed_attributes.csv`, `recomputed_latency_statistics.csv`.
- `bootstrap_rgbt234.json` and `bootstrap_rgbt234_replicates.csv` — exact method/seed and5000 paired results.
- `lineage_verification.json`, `source_inventory.json`, `runtime_boundary_diagnosis.json`, `numpy_runtime_curve_differences.csv`.
- `input_hashes_before.json`, `input_hashes_after.json`, `input_integrity.json` — all2234 input files /23198030 bytes match before, after and the supplied initial SHA mapping.
- `execution_receipts.json`, numbered command/stdout/stderr/exit receipts, and `trace/` full reviewer request/response metadata; failures are retained. Final artifact validation caught an auditor-written out-of-range `fit_complete.json:66` reference, corrected to the actual seed field at line61. The earlier report/source/trace version remains in `attempt02_report_reference/`; no experimental input or numerical result changed.

Input files, repository, datasets, Desktop handoff, production reports and environments were not modified. No SSH/GPU/neural/package-install operations were performed. Output contains scores, curves, statistics, hashes and references only.
