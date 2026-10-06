Source review PASS — candidate-relation native recovery

Reviewed 2026-10-06T18:02:02.236732+08:00 by /root/selected_native_recovery_review. Configured model gpt-6-astra, reasoning max, fresh context; configuration confirmed by parent, runtime backend independently unattested. Independence: same-family. Acceptance: provisional. Scope: COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE.

There are no unresolved source blockers in the current implementation. This is a source-only review, not runtime or accuracy acceptance. Goal remains ACTIVE_UNMET.

Two issues were fixed before this verdict:

- The recovery initially used the collector root `/data/wangwj/dataset/LasHeR`. Archived full98 inference requires `/data/wangwj/dataset/LasHeR/traingset`. Current recovery reads the original selected inference configuration and uses its actual root for inventory and evaluation.
- The recovery initially reproduced only part of the original process environment. It now imports the existing training orchestrator's environment helper, preserving device order, CUBLAS workspace, caches and thread settings directly.

The archived evidence confirms four complete 60-epoch/480-step fits and eight complete 98-sequence/49,418-frame evaluations. The originally selected model is relations_lr5_best, epoch 5. Development IoU is 0.7444150380548322, below gross_parent 0.7461261775861118 and above GOLA 0.7432785069707898. The original controller skipped new native evaluation and deleted all 20 own weights, including the selected one. These historical receipts are preserved.

The minimal orchestrator fix now always evaluates the selected trained model and retains it after native consumers, including negative runs. Parent-IoU improvement is only recorded, never used to suppress either formal benchmark.

The restoration uses the original ordered nine caches, batch 336, seed 42, initial parent, frozen C1, lr 1e-5, weight_decay 1e-4, threshold 0.03 and oracle/budgeted/action/gross settings. The unchanged trainer uses constant-LR AdamW; total epochs only bounds the loop and final metadata. GPU 1 and the original environment are reused. Thus five epochs is the original 40-update prefix in source, subject to actual runtime parity. Archived epoch 5 is the global utility best and also the best of epochs 0–5.

Before native evaluation, the runner requires epoch 5/40 completion, strict reload, A/B/C gradients and changes, changed relation parameters and frozen C1. It then compares all six metric records within 1e-6, every saved best-validation array exactly, all 98 original prediction files byte-for-byte, and nine required decision fields exactly. It makes no claim of original parameter-byte equality. No new architecture, hyperparameter choice, seed search or native-data tuning is introduced.

The unchanged native engine accepts this scope and labels the candidate selected_ABC. One restored checkpoint is used for all 245 LasHeR sequences/220,703 frames and all 234 RGBT234 sequences/116,649 frames. Its merges verify complete shard coverage and actual model/configuration. Five metrics come from the selected candidate's actual-GT report, with 19+12 attributes and metric curves. Four paired comparisons use 5000 resamples/seed 42 against GOLA, C1, old4 and gross_parent. Mechanism timelines and efficiency are collected; sequence resampling is not seed stability, and four-worker timing is identified as concurrent.

Cleanup runs only after successful full native completion for the same checkpoint. Recovery removes only its own initial.pth and last.pth within the new output root; best.pth stays regardless of scores. The protected original parent and dependencies are untouched.

The private deployment helper was also read. It uses the fixed review paths, explicit source/report staging, an empty-index guard, original-root and original-controller checks, new-output protection, disk and four-GPU preflight, and one detached launch with the shared environment. It does not query or set power/temperature. Current branch is main. The helper was not executed by this reviewer.

Validation performed: standard-library AST parse for nine pipeline source files, the deployment helper and embedded remote payload; archived four-fit/eight-full98 receipt consistency; epoch0–60 history and original best epoch; root/split/frozen-dependency consistency; and selected-weight deletion/native-skip evidence. No scientific imports, remote commands, training, inference, publication or scientific source edits were performed by the reviewer. Only these reports were written.

Execution must still establish saved-array and complete 98 parity before running full native benchmarks. Actual full native metrics and the five-metric +2 objective remain pending. The one original seed and reused developer98 do not establish seed robustness or untouched confirmation.

Primary source: scripts/run_candidate_relation_training.py; scripts/recover_candidate_relation_native.py; research/train_recoverability.py; scripts/run_selective_state_native.py. Supporting evaluator, partitioning and metric consumers are listed in the JSON report. Evidence is under refine-logs/runs/candidate_relation/complete_v2/training. Complete structured checks and line references are in the accompanying JSON.
