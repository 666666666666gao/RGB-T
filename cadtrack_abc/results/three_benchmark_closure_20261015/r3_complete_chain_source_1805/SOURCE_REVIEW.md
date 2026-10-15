# R3 真实资格检查、匹配续训对照与完整总控：源码审查

**结论：PASS_SOURCE_ONLY_WITH_NONBLOCKING_NOTE；BLOCKING 0，NONBLOCKING 1。**

三个新增文件及其已封存调用接口未发现确定的阻断源码缺陷。主路径连接正确，可以据此准备后续真实资格检查；本结论不构成 GPU、图像、训练、指标或完整依赖运行验收。

审查时间：2026-10-15T10:04:46.680892+00:00。请求模型/推理：`gpt-6-astra` / `max`；宿主没有实际路由证明，`actual_route=UNATTESTED`，实际模型/effort 不作认证。独立上下文审查仍属 `same-family`，`acceptance_status=provisional`。

**范围与输入**

输入封存目录：`C:\Users\gb\.codex_tmp\cadtrack_abc_20261013\dense_r3_chain_review_inputs_20261015`。先读取 REVIEW_SCOPE.json 和 input_manifest.json，再检查新增 qualify_dense_main.py、train_dense_continuation.py、dense_main_pipeline.py。其余30个 Python 文件和 plan 仅作为相关依赖/合同证据。

未读取服务器、NN日志、GPU状态、权重或真实图像；未 import Torch 或项目；未启动新 NN；未重算既往 CPU 证据；未改封存输入。plan 中既往审查和运行陈述未用作本次独立通过证据。

35项原始输入全部匹配 manifest，前后完全相同，manifest 自身也相同：`253c9f9c747d9f5a8488ade02b69e26d5ea63511d5ca769d4f553533dcb7eb09`（6552 bytes）。逐文件结果见 [before](input_audit_before.json) 与 [after](input_audit_after.json)。

**BLOCKING**

无确定阻断源码缺陷。尚未运行的 GPU / 真实图像检查属于运行义务，不自动记为 FAIL。

**NONBLOCKING**

N1 — `dense_main_pipeline.py:80-84`：已有子进程非零退出分支会记录阶段 `FAILED` 与真实 `actual_exit_codes`，保存后停止；但顶层 `queue_state.json.status` 仍保留第54行的 `DENSE_R3_MAIN_PIPELINE_RUNNING`，直到正常成功才在第240行修改。仅读顶层状态的观察器可能把已经停止的链显示为运行中。日志、阶段失败与退出码仍被保存，不会越过门禁，也不会写出成功终态。若修正，只需在现有失败处理处、save() 前更新顶层失败状态；不需要 retry、额外 try/except 或 fallback。本审查没有实际执行任何失败子进程。

**实际本机检查**

| 检查 | 实际结果 | 输出 / 退出码 |
|---|---|---|
| 封存输入前验 | 35/35匹配 manifest | [manifest_before](manifest_before.receipt.json)，exit 0 |
| stdlib AST解析 | 33/33通过 | [source_ast](source_ast.receipt.json)，exit 0 |
| 续训结构与CLI接口 | 11项R2结构一致；17个命令接口；缺参0、未知flag0 | [source_contracts](source_contracts.receipt.json)，exit 0 |
| 封存输入后验 | 原35项和manifest均未变化 | [manifest_after](manifest_after.receipt.json)，exit 0 |

解释器为 `E:/python.exe`（Python3.12.6）。每条 receipt 保存真实命令、开始/结束时间、stdout/stderr文件名和退出码；所有stderr文件保留（本次为空），没有检查辅助脚本失败。AST只核查本机语法，不能说明目标环境 import/库接口或运行通过。[source_contracts.json](source_contracts.json) 保存逐项结构与参数结果，[source_ast_and_cli.json](source_ast_and_cli.json) 保存33个文件的CLI定义。

**已核对的生产路径**

- **C01 — R2 terminal gate and HA overall parent**：The new controller and qualifier consume the completed-pair status and all20-order witness, require full98 selection and strict reload, and use HA best_checkpoint/selected (overall winner, including epoch0), not trained_best. Raw training also checks the HA complete report set. The original pair controller and actual terminal artifacts are outside this snapshot; their real completion is not established by this review. 依据：`dense_main_pipeline.py:31-42`；`qualify_dense_main.py:49-59`；`train_dense_motion.py:51-65`；`select_dense.py:49-64`。
- **C02 — 24 real TRAIN-clip parent parity checks**：Two seeded TRAIN clips each advance 12 real image queries through both native parent and zero-residual R3. Assertions compare native boxes, confidence, geometry/keep choice, template write decisions, template values/strides and memory. Both trackers retain the shared native Python-double crop path. The qualifier requires an actual requested template-write event; whether the fixed real clips satisfy it awaits execution. These are sampled clips, not full videos. 依据：`qualify_dense_main.py:63-105`；`dense_main_tracker.py:18-40`；`dense_main_tracker.py:60-76`；`dense_main_tracker.py:102-132`；`dense_tracker.py:35-79`；`pure_baseline_tracker.py:27-39`。
- **C03 — Six native actions, quarantine, clock and restore**：Each of the six action modes starts from the saved frame4 state. Defer advances age while preserving accepted history, frames, validity, quality, memory and templates; the second deferred step checks an increasing observation gap/deviation. Native restore checks state equality and separate history storage. The later 210-action collector additionally exercises all candidate/action combinations. 依据：`qualify_dense_main.py:129-162`；`dense_submission.py:70-81`；`dense_main_model.py:117-146`；`dense_main_tracker.py:141-148`；`dense_consequences.py:14-22`。
- **C04 — Legal independent extra contexts and native counterfactual replay**：The four contexts are local5 and three separate local5+one-extra5 alternatives. Every extra is generated from the original local query. Deployment observes at most one extra. Counterfactuals restore one prior snapshot for each of (5+10+10+10)*6=210 actions, then continue with tracker.track and the same frozen actual policy. The chosen replay is compared with native track(), and the real native successor is restored after collection. Future GT enters the label adapter, not the observation features. 依据：`qualify_dense_main.py:109-128`；`qualify_dense_main.py:159-162`；`dense_main_model.py:94-106`；`dense_main_tracker.py:92-99`；`dense_consequences.py:45-104`。
- **C05 — Inference-cache boundary, real-label head updates and strict reload**：ordinary_copy save/load is called after leaving the outer inference_mode block. Two separate newly constructed head groups load the frozen source state, consume the real evidence/consequence labels, take two AdamW steps each, require finite positive aggregate gradient norms and changed tensors, and strictly reload. Parent/tracker parameter hashes are checked afterwards. Actual tensor/autograd behavior has not been executed here. 依据：`qualify_dense_main.py:30-35`；`qualify_dense_main.py:163-196`；`train_dense_evidence.py:40-70`；`train_dense_consequences.py:48-78`。
- **C06 — Probe isolation from formal initialization**：Qualification heads are temporary; its stored artifact is a query cache. Probe and formal raw/control outputs use separate directories, and both formal entries construct a new model and reload the locked parent with a fresh optimizer/scheduler. No probe checkpoint is supplied to formal training. 依据：`qualify_dense_main.py:168-205`；`dense_main_pipeline.py:86-117`；`train_dense_motion.py:60-83`；`train_dense_continuation.py:72-87`。
- **C07 — Matched R2 A/localizer continuation objective and training budget**：The control uses DenseCADTrack(..., True), the original R2 forward/loss and optimizer loop, not the B/C raw objective. Eleven AST comparisons pass after substituting the old args.memory=True, including the entire epoch loop. Production commands explicitly give both existing localizer/prompt groups lr1e-5; new R3 temporal groups use 1e-4. Both runs use seed42, independent DataLoader generator42, the same batch, workers4, 20*20000 clips and 8 queries/clip, fresh AdamW/StepLR. The controller checks all20 actual order witnesses and parent identities after training. Actual order equality remains pending. 依据：`train_dense_continuation.py:62-101`；`train_dense_continuation.py:116-193`；`train_dense.py:58-180`；`dense_main_pipeline.py:86-99`；`dense_main_pipeline.py:117-133`；`train_dense_motion.py:85-106`。
- **C08 — Raw/control full20 then full98 positive-epoch selection and all3 tests**：The controller waits for both full20 fits and prescribed counts. Raw evaluation uses the six scheduled checkpoints [0,1,2,5,10,20], full98 selection, and trained_best for all3. The R2-family control separately evaluates the same six epochs with its existing validator/selector, and tests the same selected positive checkpoint on all3. No metric-gain condition skips or replaces negative reports. The underlying R3 video helper and official score implementation are outside this seal. 依据：`dense_main_pipeline.py:117-183`；`evaluate_dense_main.py:27-118`；`select_dense_main.py:24-69`；`select_dense.py:23-68`。
- **C09 — Two-card fresh evidence/consequence collection and merged paths**：Each split is partitioned by sorted complete sequence names [shard::2]. Two children run on physical device indices2/3, each writing its own shard directory; their records are merged as shardN/<file>. Both dataset readers use root / record[file], so the rewritten paths reach actual shard files. Receipt split/checkpoint/coverage/uniqueness are checked before merging. The evidence teacher is raw trained_best; consequences are newly collected from selected calibrated weights. The inputs reused by this controller are query JSON timestamps/specs, not old feature or action-label caches. 依据：`dense_main_pipeline.py:185-225`；`collect_dense_evidence.py:32-94`；`collect_dense_consequences.py:35-95`；`train_dense_evidence.py:18-33`；`train_dense_consequences.py:22-41`。
- **C10 — Calibration20 then current-teacher consequence policy20**：Calibration completes20 epochs, selects by declared cache-quality MSE, and strictly reloads before fresh consequence collection. Policy training runs20 epochs using future-action and selector-realized search losses and checks both decision subgroups changed; cache validation remains diagnostic. The production chain does not substitute those cache scores for final full-video selection. 依据：`dense_main_pipeline.py:215-232`；`train_dense_evidence.py:127-162`；`train_dense_consequences.py:135-197`。
- **C11 — Final selected positive checkpoint, three datasets and seven metrics**：The final fit again routes through all six full98 checkpoint evaluations and the trained positive-epoch selector. The same selected checkpoint/hash is used for LasHeR245/220703, RGBT234234/116649 and RGBT210210/104706. Reports expose each benchmark delta and +2 flag. The +2 aggregate is reported without being asserted as a completion gate; goal_achieved remains false and component effectiveness is explicitly unestablished. Actual metric formulas in cadtrack_abc.score are outside this seal. 依据：`dense_main_pipeline.py:233-246`；`evaluate_dense_main.py:68-118`；`infer_dense_main.py:25-55`；`score_dense_main.py:20-65`。
- **C12 — Command flags, fields, serialization and child lifetimes**：Seventeen assembled controller command sites match their sealed argparse interfaces (no missing required or unknown flags). The produced/consumed stage strings and selected-checkpoint/report field names agree. New JSON receipts use Python scalars/lists/dicts; tensor/NumPy query data is saved in .pt, not placed in qualification JSON. The parent waits for all children in each phase before the next phase, and the nested evaluator assigns its NN children to2/3. A nonzero child exit records phase failure/codes then stops the chain; N1 describes the stale top-level status. 依据：`dense_main_pipeline.py:64-84`；`qualify_dense_main.py:100-103`；`qualify_dense_main.py:186-207`；`dense_main_pipeline.py:203-212`；`evaluate_dense_main.py:108-120`。

**覆盖边界**

本封存未包含 `dense_main_evaluation.py`、`model.py`、`score.py`、vendor代码和原R2 pair/pipeline写出器，也没有真实权重、标签、视频、终态报告。可核查到新入口与所供解析器/字段/相对路径的衔接；无法在此独立验证未提供实现的追踪循环、原始官方指标公式、目标环境导入和原R2实际完成情况。这是覆盖边界，并不表示部署中缺文件或已经运行失败。

**仍需真实执行的义务**

1. Wait for and inspect the genuine original R2 pair terminal/all20-order/full98/all3 artifacts before the new launch. Verify the authorized physical2028 host and available GPUs2/3 through the execution owner; this review made no server/resource request.
2. Execute the finite native qualifier on the real locked parent and TRAIN images. Require all24 parity checks, actual requested template-write coverage, six-action state/clock/restore checks, four legal contexts, 210 native consequences/replay, two head-group real-label AdamW checks and strict reload. Pending is not a FAIL.
3. Execute the two raw/control probes, then separate full20 raw/control fits with fresh parent initialization. Inspect actual updates, all20 data-order witnesses, gradients/changes, frozen backbone, strict reload and real terminal counts.
4. Complete six full98 evaluations and one selected positive-epoch checkpoint across all3 datasets for raw and control. Preserve negative results and the distinction between overall epoch0 and trained best.
5. Collect all requested TRAIN/DEV evidence and consequences freshly in disjoint sequence shards. Verify actual per-query files, valid labels and merged receipt/path coverage; then complete calibration20 and both future/search heads for policy20 using the selected calibrated teacher.
6. Complete final six full98 runs and all3/seven metrics with one positive checkpoint/config, then perform result/runtime audit. Pipeline completion or a +2 flag alone does not establish the three component effects or complete the research goal. The all8 component ablations are not implemented by this scope.

总控只在raw20、匹配对照20、calibration20、policy20以及对应完整full98/三集报告真正返回后写完成状态。其 `goal_achieved=False` 与组件效果未建立的说明正确。七项+2目标、独立组件有效性和结果验收仍未完成；本报告不提供新正式成绩。
