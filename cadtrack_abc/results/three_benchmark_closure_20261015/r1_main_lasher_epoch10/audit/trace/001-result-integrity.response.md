# Experiment audit: R1 selected epoch 10 LasHeR

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

The completed scope is **245 sequences / 220703 frames**, with **220458 post-initialization tracking calls**. Shard 0 contains 123 sequences / 107898 frames and shard 1 122 / 112805; the exact sorted alternating partitions, all sequence names, all shapes, finite data, all 245 first-box initializations and receipt frame/time sums agree (`predictions/complete_shard0.json:748`; `predictions/complete_shard 1.json:742`; `project/cadtrack_abc/infer.py:48`). One training seed 42 is recorded (`fit_complete.json:61`). The current plan permits one best seed and does not require three successful seeds (`documentation/plan_snapshot.json:376`).

## A. Ground-truth provenance — PASS within the supplied artifact boundary

The scorer reads actual dataset `init.txt` files (`project/cadtrack_abc/score.py:36`); it never uses baseline predictions as GT. The raw manifest identifies `/data/wangwj/dataset/LasHeR/testingset` as the source (`raw_input_manifest.json:4606`). All 245 source paths and every one of the 767 manifest file SHA256/size pairs match the collected files. GT arrays and prediction arrays are distinct; none of the 220458 noninitial prediction rows exactly equals its GT row, and no GT file byte hash equals a prediction file hash. This is corroboration against direct copying, not a universal proof against fabrication.

Inference reads only the first annotation line and then calls `model.step` with current images, frame index and state (`project/cadtrack_abc/infer.py:56`, `:65`). All serialized first boxes match the float32 GT initialization rounded to three decimals. The supplied source does not reveal later-GT-fed inference. Full no-leakage/runtime certification is unavailable because `counterfactual.py` and vendor code are absent (`project/cadtrack_abc/search_actions.py:10`, `:99`; `project/cadtrack_abc/baseline.py:7`). No neural code was executed in this review.

The original official download/archive hashes and original toolkit attribute text files are not included. Therefore the authenticity claim stops at the supplied dataset/source manifest; it is not an independent reauthentication of the official distribution.

## B. Score normalization and exact protocol — PASS

No metric denominator is the maximum, minimum or mean of model predictions. NPR uses **GT width/height + 1e-8** (`toolkit/rgbt/utils.py:118`, `:141`). The exact supplied installed-toolkit protocol is:

1. Inference serializes xywh predictions to three decimals (`project/cadtrack_abc/infer.py:86`). The toolkit loads them as float32 and applies `round(0)` (`toolkit/rgbt/utils.py:31`; `toolkit/rgbt/dataset/basedataset.py:22`). Raw GT is loaded as float64, bypassing constructor GT rounding (`project/cadtrack_abc/score.py:36`). Integer rounding changes 875755 coordinate entries.
2. Metrics execute **NPR → PR → SR** (`project/cadtrack_abc/score.py:54`). Each replaces prediction 0 with GT 0 and **retains the initial frame in its denominator**; the source comment “ignore the first frame” means replacement, not removal. Invalid predicted width/height would carry forward the preceding prediction; no supplied prediction triggers this rule.
3. PR uses center error `<= 20` pixels, with 51 thresholds 0…50. NPR uses normalized center error `<= 0.20`, with 51 thresholds 0…0.5 in steps of 0.01. The representative score is threshold index 20 (`toolkit/rgbt/metrics/metrics.py:280`, `:323`).
4. SR uses the inclusive-coordinate IoU implementation and strict `IoU > threshold`, at 21 thresholds 0…1. Its reported score is the **arithmetic mean of all 21 samples**, not trapezoidal AUC and not plain mean IoU (`toolkit/rgbt/utils.py:162`; `toolkit/rgbt/metrics/metrics.py:236`).
5. Each sequence has equal weight regardless of length. Attribute scores use the same equal weighting over exported member sequences (`project/cadtrack_abc/score.py:60`, `:65`).
6. **Native invalid-GT behavior is material:** any x/y/w/h `<=0` sets center error / IoU to `-1`. That counts as success for PR/NPR at every nonnegative threshold and failure for SR; such frames are not dropped (`toolkit/rgbt/metrics/metrics.py:266`, `:309`, `:351`). There are 1260 affected frames in 52 sequences: 533 with nonpositive dimensions and 727 with positive dimensions but nonpositive x/y. Their contribution to sequence-equal PR/NPR numerators is 0.5568053864558151 pp with original denominators. This is a description of the preserved protocol, not a new score or a correction to published benchmark conventions.

The audit preserved every rule. It does not certify that this collected toolkit snapshot is an untouched upstream official release. All 19 attribute scores match **conditional on `toolkit_protocol.json:251` membership**. The exporter reads installed evaluator membership (`export_toolkit_protocol.py:8`); original `AttriSeqsTxt` files used at `toolkit/rgbt/dataset/lasher_dataset.py:78` are absent.

## C. Result existence, arithmetic and execution provenance — WARN

Both completion receipts,245 prediction files,245 latency arrays, `full_report.json`, `per_sequence.csv`, and `mechanism_report.json` exist. Replaying the supplied metric and loader bodies, then separately implementing the equations in NumPy, yields exact equality for **30135 per-sequence curve values, 123 mean-curve values, 735 per-sequence scores in each of JSON and CSV, 57 attribute metrics and 3 overall metrics**. All threshold arrays match. There are 41963 numeric/structural/hash comparisons plus 9 static checks: **41972 comparisons, 0 failures**, maximum score/curve/latency discrepancy 0.0. See `deterministic_verification.json`, `static_verification.json`, and `verification_execution_receipt.json`.

The audit replayed original mathematical class/function bodies after separating imports, avoiding absent dataset/plotting dependencies. It did not execute the original full CLI, load checkpoints, use GPU/SSH, run neural inference, or install packages. An independent NumPy implementation agrees with every curve value. Python 3.12.6 and preexisting NumPy 1.26.4 were used. The initial import attempt lacked NumPy on its path; the genuine `ModuleNotFoundError` is retained in `audit_tool_attempts.json`. Only the review process PYTHONPATH was set to the existing cache; no environment files were changed.

Epoch 10 is the actual maximum of the six candidate summaries, and also the best positive epoch (`selection_snapshot.json:2`, `:9`, `:47`; selection rule at `project/cadtrack_abc/select_future_actions.py:42`). The best path and recorded SHA `a14130fafbc7eb49a17a3834fe8a43855f3ed0833ba576e737e74d1579965ad8` agree with the observation and plan (`main_observation_1040.json:6`; `documentation/plan_snapshot.json:853`). The fit receipt's 1600 updates equals 20×ceil(5092/64). These checks establish metadata consistency, not independent reproduction of training, DEV IoUs, optimizer state, actual loaded weights or strict reload. The weight bytes, raw DEV records and original LasHeR native process/exit logs are absent; `raw_input_manifest.json:4610` explicitly says no images/checkpoints were copied.

The relay's 14-file source seal is narrower than the active dependency graph: it omits `search_actions.py`, `ported_modules.py` and `mechanism_summary.py`; the collected bundle also lacks `counterfactual.py` and CADTrack vendor sources. Collection-time hashes must not be promoted to complete launch-time/runtime provenance (`main_formal_relay.py:26`; `static_verification.json`). No active source was changed to address this audit limitation.

### W-TIMING-01: incorrect hardware-scope text

`report/full_report.json:402` and `project/cadtrack_abc/score.py:81` say “Concurrent two-GPU run”. This relay constructs both shard commands and runs them with `gpu=2` (`main_formal_relay.py:89`, `:92`); the handoff identifies both LasHeR processes on GPU 2 (`documentation/HANDOFF_snapshot.md:5100`). Correct the derived presentation to **two concurrent inference workers on physical GPU 2**, acknowledging shared-host concurrent work. Preserve the original sealed report.

All 245 latency arrays and every per-sequence receipt sum reproduce exactly:

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

The mechanism summary can be **partially** independently checked. From raw final predictions/GT, this review exactly reconstructs 219925 valid post-initialization GT frames, 70077 failure frames, sequence-equal IoU 0.5894167200173702, frame-equal IoU 0.5120501208799806, all 881 failure/recovery events, 792 recoveries, 89 right-censored events and median recovery delay 11 frames. The diagnostic mask intentionally differs from official scores (`report/mechanism_report.json:7776`).

No raw `*_mechanisms.json` files are supplied. Candidate recall, newly recalled acceptance, same-state rescues/harms, alternative selections and commit/pollution counters require those records (`project/cadtrack_abc/mechanism_summary.py:54`, `:59`, `:79`). Their ratios and total 7380 extra forwards agree internally with receipt summaries, but that is not independent verification of per-frame behavior or actual visual execution. These unavailable checks are recorded as PARTIAL, never PASS by implication.

## E. Scope — WARN

The result supports one selected fixed-model LasHeR point estimate. PR/NPR are below the user reference, SR rises only 0.0419915 pp; none of the +6 pp targets is met. The observed controller is still running RGBT234 and has only a completed LasHeR report (`main_observation_1040.json:5`, `:9`, `:60`). This bundle supplies no completed RGBT210/RGBT234 result for this checkpoint, matched-cost experiment or effective A/B/C control result. Overall completion remains unsupported (`documentation/plan_snapshot.json:351`, `:387`, `:389`).

`fit_complete.json:53` and `selection_snapshot.json:55` retain earlier/full-pipeline completion flags. They do not negate the newer completed LasHeR report: the relay sets global formal completion only after all three datasets (`main_formal_relay.py:116`). The handoff snapshot ends before the 10:40 result observation. Historical +2/LasHeR-only plans are superseded by the explicit current acceptance section, not silently rewritten.

## F. Evaluation type — real_gt / PASS

Standard PR/NPR/SR use dataset annotations and are `real_gt`. Post-hoc mechanism diagnostics use GT-assisted localization proxies and explicitly deny semantic identity or causal future-effect interpretation (`report/mechanism_report.json:7776`, `:7778`; `project/cadtrack_abc/mechanism_summary.py:118`). They are not prediction-generated replacement GT.

## Preservation and actions

All 778 enumerated input SHA256 values agree with the initial manifest and remain unchanged before/after verification. All 767 collected source-manifest hashes and byte counts match. The 11 files outside that source manifest are separately sealed by the initial/before/after hash manifests; this includes snapshots, exporter and audit-manifest metadata, not missing raw GT/predictions. No raw GT/prediction coordinates or latency arrays are emitted in these audit outputs.

1. Correct the timing concurrency description in derived reporting; keep original evidence immutable.
2. Report the exact supplied-toolkit semantics and the limited provenance of exported attribute membership.
3. Limit validated mechanism claims to the reconstructed subset; obtain the existing raw traces before auditing other mechanism counters.
4. Preserve the selected weight/source/DEV/native receipts for any stronger lineage claim. No new neural run is requested by this audit.
5. Keep the overall objective unmet until corresponding completed benchmark, target and mechanism/cost evidence exists.

Artifacts: `EXPERIMENT_AUDIT.json`, `deterministic_verification.json`, `static_verification.json`, `verified_per_sequence.csv`, `verified_per_sequence_curves.csv`, `verified_attributes.csv`, `input_hashes_before.json`, `input_hashes_after.json`, `source_manifest_verification.json`, `verification_execution_receipt.json`, and `trace/`. Deterministic arithmetic acceptance does not upgrade this same-family semantic review from provisional status.
