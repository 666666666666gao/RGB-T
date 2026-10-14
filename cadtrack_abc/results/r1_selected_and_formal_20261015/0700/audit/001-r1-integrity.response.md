# CADTrack ABC R1 experiment integrity audit

**Overall: WARN — 0 blocking issues.** Completed main full98 selection passes exact deterministic checks. R1 formal three-benchmark performance and overall goal achievement remain unproved.

Date: 2026-10-15. Generated: 2026-10-14T23:16:15.661886+00:00. Reviewer: fresh independent-context native Codex agent, same-family/provisional. Requested gpt-6-astra / max; actual model and effort routing **UNATTESTED**.

Read all 37 supplied files (16 inventory-listed plus 21 additional), with no missing input or unresolved locator. 57 files were hashed after targeted dependency/evidence inspection and all remained unchanged. No SSH, GPU query, NN forward, optimizer update, download or environment change.

## Deterministic completed-artifact result

2598 standard-library CPU checks passed; 12 shards and 588 candidate/sequence records checked. The manifest gives exactly 881 TRAIN and 98 disjoint DEV names. Each candidate has the exact expected 49/49 sorted shard names and 52,200 frames (20,746/31,454), with 52,120 valid GT frames (20,746/31,374). Per-sequence frame counts also match the separately recorded previous full sampling inventory.

| Epoch | Sequence-equal mean IoU | Frame-equal mean IoU | Extra forwards | Future interventions |
|---:|---:|---:|---:|---:|
| 0 | 0.6498416023214861 | 0.4973244761103223 | 0 | 0 |
| 1 | 0.6576578945909836 | 0.4879377006478064 | 0 | 575 |
| 2 | 0.6457127683358838 | 0.5294795227343544 | 0 | 1408 |
| 5 | 0.6561556810855257 | 0.5630953458484466 | 0 | 474 |
| 10 | 0.6664379662549009 | 0.5816733049962124 | 1177 | 1028 |
| 20 | 0.6638382770577256 | 0.5594530161032746 | 5703 | 1291 |

All 12 reported aggregate values are bit-exact matches to independent Python recomputation. All 588 row means exactly match float32(iou_sum / valid_frames). Epoch10 is both the overall winner and the best positive-epoch checkpoint. Its sequence-equal development gain over epoch0 is 0.016596363933414793 (1.6596363933 percentage points of this DEV IoU metric). This is not official SR or evidence for an isolated module contribution.

The fit receipts record 20 epochs and 1,600 updates each, consistent with 20 × ceil(5,092/64). Head parameter counts independently recompute to 37,426 for future C and 39,252 additional for search B, totaling 76,678 for the joint head fit. Native ABC/base/motion coordinates are outside this optimizer.

Remote selected checkpoint SHA recorded at admission: `a14130fafbc7eb49a17a3834fe8a43855f3ed0833ba576e737e74d1579965ad8`. This is a receipt assertion, not a locally recomputed checkpoint hash. Local config raw SHA `92c673aa715afa69c0ea02220e2db9c6335c6dfd8ed60c775ffabe61e8807de9` differs only by CRLF; its LF bytes exactly reproduce admitted remote SHA `409a41e510dcccaa81e055f5e2eab07b7fa1ad3822e3e02ca018841c4dc701e6`.

Detailed checks: `deterministic_verification.json`; all raw receipt rows: `all_588_candidate_sequence_records.csv`; input hashes: `audited_input_hashes.json`.

## A_ground_truth_provenance: PASS

Source provenance and completed receipt fields; raw GT and prediction arrays not supplied.

- Development loads dataset init.txt, initializes only from frame0, calls step without annotations, then computes IoU against the dataset labels after the continuous run.
- Formal inference reads only the first annotation line; formal scoring independently loads actual dataset GT. No enhanced or baseline output is substituted for GT.
- Counterfactual action supervision uses real current and future dataset boxes in label_rollout; model rollouts are predictions. The wrong-commit indicator is explicitly a localization proxy, not semantic identity truth.
- RGBT210 receipt has 210 unique sequences and 104706 paired frames; all 210 stored per-sequence GT comparisons agree with the installed toolkit inventory. This audit did not reopen remote images/GT or the archive.

Evidence:

- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:55`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:58`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:61`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:65`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:69`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:80`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\infer.py:56`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\infer.py:58`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\infer.py:60`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\infer.py:65`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:35`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:38`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:39`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:53`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:55`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:65`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:68`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:75`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:89`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:92`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:102`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:127`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:130`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\batched_actions.py:110`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\batched_actions.py:113`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\counterfactual.py:140`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\counterfactual.py:150`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\counterfactual.py:156`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\counterfactual.py:165`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\extract_verify_official_rgbt210_remote.py:44`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\extract_verify_official_rgbt210_remote.py:47`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\extract_verify_official_rgbt210_remote.py:49`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\extract_verify_official_rgbt210_remote.py:50`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\extract_verify_official_rgbt210_remote.py:53`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\extract_verify_official_rgbt210_remote.stdout.txt:5`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\extract_verify_official_rgbt210_remote.stdout.txt:6`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\extract_verify_official_rgbt210_remote.stdout.txt:8`

## B_score_normalization: PASS

Project metric and aggregation code; archived toolkit excerpt, not fresh attestation of every imported utility.

- Development uses ordinary IoU intersection/union and sequence-count or valid-frame-count denominators. All 588 float32 record means exactly equal float32(iou_sum / valid_frames).
- Official wrapper multiplies toolkit values by 100 only. Archived PR/MPR/SR/MSR code divides by frame counts and averages sequence curves, not prediction maxima, means or minima.
- Runtime candidate scores, probability softmax, geometric normalization and relative action utility are model/learning computations, not reported accuracy normalization.
- The archived NPR class calls normalize_CLE with prediction and GT; the body of that external utility is not present in the inspected archive, so no independent claim is made about its full implementation.

Evidence:

- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\data.py:51`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\data.py:57`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:39`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:41`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:54`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:55`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:57`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:63`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:70`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:40`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:43`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:75`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:79`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:192`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:196`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:228`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:232`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:349`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:357`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_metric_implementation.stdout.txt:360`

## C_result_existence_and_numbers: PASS

All supplied files exist; completed main selection is exactly recomputed. Formal R1 results remain absent/pending, as declared.

- Read every one of the 16 inventory-listed files and all 21 additional requested inputs. No missing input or unresolved locator error.
- All six epochs 0/1/2/5/10/20 have both full98 receipts. Exact manifest names, sorted shard membership, policy, checkpoint path, epoch, frame scope and flags match.
- All 12 aggregate numbers are bit-exact Python float matches to selection.json. Overall and positive-epoch selection both select epoch10.
- Both completed fit receipts agree with the terminal verification record. Parameter arithmetic is 37426 future-head, 39252 search-head, 76678 joint; 20 * ceil(5092/64) = 1600 updates.
- Historical native LasHeR and pure CAD RGBT234 numbers in plan_snapshot have existing full reports with matching metric values. This is existence/number consistency, not a fresh recomputation of those historical benchmarks.
- No R1 full_report.json or completed R1 three-dataset score is supplied. Both selection and admission explicitly keep formal completion false.

Evidence:

- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:2`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:9`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:47`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:54`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:55`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:99`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:100`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:101`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_future_complete.json:2`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_future_complete.json:4`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_future_complete.json:7`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_future_complete.json:8`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_future_complete.json:10`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_future_complete.json:11`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_future_complete.json:46`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_future_complete.json:48`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_search_complete.json:2`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_search_complete.json:4`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_search_complete.json:7`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_search_complete.json:8`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_search_complete.json:10`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_search_complete.json:11`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_search_complete.json:52`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\fit_search_complete.json:54`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\results\r1_actual_admission_20261015\full_fit_terminal_verified.json:1`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:112`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:115`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:135`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:141`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:148`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:155`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:175`

## D_metric_call_paths: PASS

Completed development receipts and static pending formal call graph; not a claim of completed formal execution.

- Development IoU is invoked on predictions and GT and written into each completed shard. The selector consumes those exact fields.
- cached_validation is called after each fitting epoch and explicitly emits formal_metrics=null; it is not the selection rule.
- Both relay scripts invoke cadtrack_abc.score after successful inference and receipt checks. score registers the tracker and calls dataset metric methods, then serializes the values and per-sequence/curve outputs.
- LasHeR mechanism helpers are reachable through score -> summarize_lasher -> box_iou_xywh/recovery_events/ratio. No advertised project metric is only defined and never wired into the relevant pipeline.
- Archived toolkit classes bind LasHeR PR/NPR/SR and RGBT234 MPR/MSR to the shown implementations. Imported toolkit runtime and new R1 formal outputs are not re-executed by this audit.

Evidence:

- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:69`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:72`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:83`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:33`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:39`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:41`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:35`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:60`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:157`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:102`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:104`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:154`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:156`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:53`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:55`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:93`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:94`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:97`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\mechanism_summary.py:58`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\mechanism_summary.py:59`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\mechanism_summary.py:81`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\mechanism_summary.py:95`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\mechanism_summary.py:102`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:21`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:23`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:104`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:118`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:139`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:204`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:205`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:286`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\read_toolkit_classes.stdout.txt:319`

## E_scope_and_claims: WARN

- Completed: one seed42, two 20-epoch head-only fits over 5092 cached TRAIN queries from 881 sequences, with 605 DEV cached queries. Main search_actions six-candidate continuous evaluation covers the same 98 repeatedly used DEV videos, 52200 frames / 52120 positive-width-height GT frames per candidate.
- The six candidate evaluations are six checkpoints of one run, not six seeds, independent test sets or a six-run statistical sample.
- Development mean IoU includes the initialization frame and uses positive width/height masking. It is not official LasHeR SR; archived official toolkit uses native rounding and its own invalid-GT handling.
- R1 freezes native ABC/base/motion coordinates and trains only future_action, plus search_action for main. This does not establish retraining or independent efficacy of all three full modules.
- No R1 formal three-dataset metrics, +6 result, paired-baseline uncertainty, baseline-matched speed improvement or overall goal completion can be certified from this snapshot.
- Some plan_snapshot fields are stale relative to later admission/selection receipts; see finding W2.

Evidence:

- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:87`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:97`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:100`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:112`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:115`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:126`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:178`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:57`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:59`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:68`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:80`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:55`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:56`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:79`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:81`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:82`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:85`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\score.py:92`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:363`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:375`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:378`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:387`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:389`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:395`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:397`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:518`

## F_evaluation_type: PASS

- Counterfactual utility is a training/development diagnostic from frozen model trajectories scored against real annotations, not an official benchmark or model-generated reference.
- Wrong identity/template/geometry commits are localization-IoU proxies; semantic identity error labels are not established.
- None of the inspected evaluation paths is human_eval, self_supervised_proxy or simulation_only. No unlabeled synthetic-reference metric was found.

Classification: {"completed_full98_development": "real_gt", "cached_counterfactual_utility": "real_gt", "future_formal_benchmarks_if_executed_as_written": "real_gt", "mechanism_localization_diagnostics": "real_gt"}

Evidence:

- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\counterfactual.py:150`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\counterfactual.py:156`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\counterfactual.py:165`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:129`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:130`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\collect_actions.py:131`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:55`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:69`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:80`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\mechanism_summary.py:117`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\mechanism_summary.py:118`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\mechanism_summary.py:119`

## Findings and required claim limits

### W1 — WARN, nonblocking: Completed numerical audit is receipt-level; formal R1 performance remains pending.

The 2598 deterministic checks verify all supplied completed main DEV aggregates and record identities. Raw per-frame predictions/GT and candidate checkpoint bytes are absent from the supplied R1 snapshot, so the underlying 313200 frame-overlap computations and checkpoint tensor equality were not independently rerun. Admission proves a recorded start only, not completion or accuracy.

Action: Keep claims at completed fit / full98 selection scope. At actual closure, use original child exit status plus the required inference/score receipts and preserve their hashes; do not substitute this audit for formal results.

Evidence:

- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:54`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:55`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:7`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:99`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:100`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:101`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:69`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:70`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:71`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\validate_future_actions.py:83`

### W2 — WARN, nonblocking: Plan snapshot contains stale current-status fields.

plan_snapshot current_acceptance says main final selection is not yet run and control has not started; its R1 status also says full98 main started and older configuration says one worker per card. Later raw selection/admission record main epoch10 selected, main LasHeR started, control epoch00 admitted, and two workers/card. Older native and baseline results are separately present and are not R1 results.

Action: Use the newer immutable receipts for current claims and reconcile status prose at the next normal publication. No input was altered in this audit.

Evidence:

- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:395`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:397`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:491`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:501`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:715`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:741`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:3`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:54`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:95`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:99`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:100`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\record_r1_control_epoch00_admission_remote.stdout.txt:5`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\record_r1_control_epoch00_admission_remote.stdout.txt:7`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\record_r1_control_epoch00_admission_remote.stdout.txt:36`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\record_r1_control_epoch00_admission_remote.stdout.txt:38`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\record_r1_control_epoch00_admission_remote.stdout.txt:84`

### W3 — WARN, nonblocking: Same-configuration and source identity are only partially pinned for future phases.

Main rehashes selected checkpoint and config before each dataset, and all 14 admission-listed source hashes match local files. Its list omits runtime dependencies including search_actions.py, counterfactual.py, ported_modules.py and mechanism_summary.py. Control rehashes each selected/trained checkpoint per dataset and passes one config path throughout, but has no analogous config-content or complete-source guard. No actual mutation is evidenced; this is a limit on guarantees, not a demonstrated runtime violation. Local config raw SHA differs from admission only because of CRLF: converting to LF exactly reproduces 409a41e510dcccaa81e055f5e2eab07b7fa1ad3822e3e02ca018841c4dc701e6.

Action: Do not claim a fully attested immutable runtime from this static audit. No speculative defensive-code change or restart of current work is required by this finding.

Evidence:

- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:19`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:26`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:29`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:71`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:72`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:73`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:89`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:15`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:16`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:129`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:130`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:133`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:144`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:8`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:9`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_main_reports_20261015\admission.json:23`

### W4 — WARN, nonblocking: State JSON alone does not establish liveness or successful completion.

Both relay run functions save COMMANDS_EXITED_ZERO before artifact validation. A missing/incorrect receipt raises without retry; top-level status may remain RUNNING and a phase may remain COMMANDS_EXITED_ZERO. Child nonzero exits mark that phase FAILED and assert; there is no automatic retry. This is fail-stop dependency behavior, but terminal diagnosis needs actual process exit plus the final receipt set.

Action: Interpret RUNNING/COMMANDS_EXITED_ZERO only as intermediate state; require original terminal exit and verified receipts at the scheduled intake.

Evidence:

- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:58`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:61`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:64`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:93`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:98`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:120`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:53`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:56`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:59`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:65`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:102`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:120`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:169`

### W5 — WARN, nonblocking: Development improvement does not isolate search B or prove three effective modules.

Epoch10 raises sequence-equal DEV IoU by 0.016596363933414793 over epoch0, and emits 1177 extra forwards with 1028 future interventions. Future and search heads train jointly, native ABC is frozen, and control full98 selection/formal results are pending in the supplied snapshot. These observations do not isolate the causal gain of B, C, or all three modules; one seed42 and repeatedly used DEV also do not establish multi-seed robustness.

Action: Report joint-head DEV selection only, with head-only/frozen-teacher scope and the full future control comparison still pending.

Evidence:

- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:13`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\outputs\r1_matched_fits_20261015\fit_search_actions\selection.json:37`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:24`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:30`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:112`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:115`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:126`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\train_future_actions.py:178`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:387`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\r1_integrity_inputs_20261015_0705\plan_snapshot.json:518`

## Relay source audit

Static pending-phase inspection; main admission and control epoch00 admission are historical supplied evidence, no live query performed.

### main

Physical GPU 2; at most 2 own workers per phase/card.

- Reads existing full98 selection and best.pth. Does not repeat sampling, collection, fitting or completed six-candidate development.
- Requires epoch10 and selected==trained checkpoint; same checkpoint digest/config digest are checked before each dataset.
- Every run joins all children and asserts all exit0; inference requires completed dataset/policy/epoch/checkpoint and sequence/frame counts; scoring requires completed correct dataset/epoch/checkpoint, scope and exact metric keys.
- The single selected epoch10 is already a positive-epoch trained model, so a separate duplicate trained role is unnecessary.
- No LasHeR user-baseline rerun is invoked.

Evidence:

- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:12`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:15`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:16`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:48`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:58`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:64`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:70`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:71`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:72`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:89`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:91`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:92`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:93`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:102`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:104`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:116`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:117`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_main_three_benchmarks_remote.py:122`

### control

Physical GPU 3; at most 2 own workers per phase/card.

- Waits until original epoch00 child exits are recorded; requires both child_exit_code==0 and full98 receipts. Evaluates only fitted candidates[1:] once; no recollection or refitting.
- Fixed future_actions policy and same config path; each selected/trained checkpoint digest checked before each dataset. Config content is not separately pinned (W3).
- Original epoch00 exit receipts plus exact full98 name/shard/checkpoint/policy/frame checks; all new child exits0; all formal receipt/report scopes and metric keys checked.
- Selection computes best positive-epoch trained candidate. If its path differs from selected best, both selected and trained roles run all three datasets before formal completion becomes true.
- First scheduled node 2026-10-15 07:12 BJT; 240-second necessary rechecks of original jobs; monitor loop does not retry evaluation.
- No LasHeR user-baseline rerun is invoked.

Evidence:

- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:18`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:20`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:21`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:45`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:53`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:59`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:61`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:70`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:72`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:79`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:82`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:101`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:102`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:103`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:111`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:115`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:119`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:122`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:126`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:127`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:128`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:132`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:133`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:144`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:146`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:148`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:154`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:157`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:165`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:166`
- `C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\run_r1_control_remaining_and_three_benchmarks_remote.py:170`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:42`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:43`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:54`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:55`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:56`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:58`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:61`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:65`
- `C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\cadtrack_abc\select_future_actions.py:66`

No retry branch in either relay. Output root creation uses exist_ok=False, so these one-shot relay programs must not be restarted over their current output roots. Artifact assertions stop dependent phases; see W4 for state freshness limits.

Code has at most two own children/card per phase and joins before next phase. Supplied admission records exactly two workers on each of physical2/3 at its timestamp. Future external competing processes were not observed or certified.

Both state records explicitly set goal_completion_not_claimed; final *_COMPLETE concerns benchmark coverage, not target achievement.

## Claims

| Claim | Status | Exact scope |
|---|---|---|
| Main R1 completed six full98 development candidates and selected epoch10. | supported | Receipt identity, exact frame scopes and numerical aggregation verified deterministically. |
| Selected main DEV sequence/frame mean IoU is 0.6664379662549009 / 0.5816733049962124. | supported | Repeated98 development; 52200 total / 52120 valid frames; not official SR. |
| Both R1 head fits completed 20 epochs and 1600 updates from the same teacher and seed42. | supported_with_qualifier | Existing terminal/fit receipts and source/arithmetic agree; optimizer tensor contents and full training logs not independently replayed. |
| All three core modules/backbone were retrained in R1. | unsupported | R1 optimizer contains only future head, plus search head in main; native ABC/base/motion coordinates are frozen. |
| Main R1 formal LasHeR/RGBT210/RGBT234 metrics or +6 targets are achieved. | unsupported | New formal metrics are pending in snapshot; startup/admission cannot substitute for scores. |
| B search activation is observed for epoch10 on DEV. | supported | 1177 recorded extra forwards; independent benefit is not proven. |
| Three independent effective contributions, multi-seed robustness, better speed or paired baseline gain are established. | unsupported | One seed, joint-head main, pending control outcomes; no paired user-baseline rerun or matched timing. |
| Both relay sources preserve the selected/trained reporting roles, two-own-workers/card and failure-without-retry design. | supported_with_qualifier | Static source and past admission only; future runtime and all dependency bytes remain unverified. |
| RGBT210 extraction inventory reports all210 sequences/104706 frames and GT equality to installed toolkit. | supported_with_qualifier | All inventory rows/sums agree with extraction stdout/source; archive/raw GT were not reopened locally. |

## Limitations

- Fresh independent context, same model family. Requested gpt-6-astra/max; actual model/effort routing is UNATTESTED by this agent.
- No SSH, download, environment change, GPU/process query, neural inference, optimizer update or source/input modification was performed.
- All completed receipt arithmetic was independently recomputed with Python standard library. It is not a recomputation of IoU from per-frame raw boxes and actual remote GT arrays.
- Remote checkpoint SHA a14130fafbc7eb49a17a3834fe8a43855f3ed0833ba576e737e74d1579965ad8 is reported by the admission artifact; no checkpoint bytes were supplied/read locally for independent SHA or tensor verification.
- Head training completion and frozen dependency values rely on receipts plus audited source wiring; no optimizer or checkpoint tensor replay was authorized.
- Archived toolkit source excerpts were inspected; runtime version and every utility implementation were not freshly attested.
- Control completed full98 receipts and R1 formal reports are outside the supplied snapshot. Any later runtime progress is intentionally unobserved.
- Semantic verdict provisional; deterministic arithmetic accepted only within its stated inputs/scope. No future runtime or overall objective achievement is certified.
