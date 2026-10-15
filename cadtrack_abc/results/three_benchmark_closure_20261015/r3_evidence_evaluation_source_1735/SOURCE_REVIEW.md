# R3 evidence/evaluation source review

**WARN — 0 blocking、1 non-blocking。** 在封存版新增八个入口及其实际依赖接口中，未确认阻断性源码缺陷。此结论仅限源码；不代表真实视频/GPU资格、训练完成或全 ABC 达成目标。

请求模型/推理强度：`gpt-6-astra / max`。实际路由、模型和强度均为 **UNATTESTED**（无 host 证明）；`review_independence=same-family`，`acceptance_status=provisional`。

## 唯一发现：R3-EVAL-N1（LOW，NON-BLOCKING）

R3 qualified 报告仍链接带旧训练标签定义的 raw mechanism report。

实际路径：`score_dense_main.py:25–29` 调用原 `score.py`；LasHeR 分支在 `score.py:93–96` 必写 `summarize_lasher`。`dense_main_tracker.py:131` 把未裁剪的 `query['pair']['boxes']` 写入 `candidate_boxes_xyxy`，`mechanism_summary.py:79–87` 据此计算 candidate recall；其第 114 行却称这些框是训练标签使用的框。新增 `collect_dense_evidence.py:70–71` 实际以 `query['native_boxes']` 计算质量标签，native clipping 见 `dense_main_tracker.py:38–40`。

因此报告中的训练标签口径说明不适用于 R3。字段本身兼容，官方七项指标、checkpoint 选择和校准标签均不受影响；没有运行数据，不能声称已观察到具体 recall 数值偏差。

**最小修复：** 仅在 `score_dense_main.py` 生成的 qualified report 增加明确 scope：所链接 raw mechanism 的 recall 按未裁剪 pair proposals 统计，R3 evidence 质量标签按 native clipped boxes 计算，二者口径不同。保留原 scorer、raw report、指标值及活跃 R2 文件。本审查未改源码，发现仍属于当前封存版。

## 已核对正确的主链

- 采集从 frame0 首 GT 初始化，以预测推进所有 prefix 帧；查询 GT 只用于 native-box overlap 标签。三次 extra 均从同一 local query 构造，形成 local5 或 local5+独立 extra5 的四个 context；没有未来图像输入。
- 校准分开编码 local/source0 与 extra/source1，再做 5/10 候选关系计算；候选、appearance、distribution 与 `[batch,candidates]` 标签的拼接顺序一致。优化器仅含 `DenseCandidateEvidence`，保存时其余 parent visual/A/motion/decision 权重直接保留。
- 校准保存 epoch0 和各训练 epoch，按辅助 cache MSE 选择，并 strict reload 所选 candidate 状态。保留 epoch0 获选的可能，显式不把初始化或 MSE 当作 tracking 增益。
- 新 loader 复用严格嵌套 state loading、pretrained/threshold 核对及原 `track_video`。验证仅首 GT 初始化；98 个唯一序列、52200 帧、split 及 `[0,1,2,5,10,20]` 六快照的选择接线一致，overall 和 positive epoch 分开保存。
- 控制器向三套数据传入同一个 positive checkpoint；245/220703、234/116649、210/104706 的总量与计划一致。复用原官方 scorer 和既有 baseline，+2 判断覆盖实际七项指标，stage/training flags 保留。所有子进程通过 `wait()` 取得真实退出码；GPU worker 仅用物理 2/3，CPU 选择/评分隐藏 CUDA。
- mechanism consumer 所需字段齐全；正式预测采用 tab/三位小数，开发预测采用 tab/17 位有效数字。报告始终保留 `goal_achieved=False`，没有将阶段完成等同于模块有效。

## 检查记录与边界

22 个封存输入均与 manifest 一致，manifest 与全部输入前后不变。新增八个文件 AST 通过；另以 stdlib/AST 核对诊断字段和序列化常量。未导入 Torch/NumPy/cv2/项目，未执行项目代码、SSH/network、NN/GPU、进程或日志观察，未安装、使用 Git、改源或派生 agent。

真实 stdout/stderr/exit 保存在 `before.*`、`interface.*`、`after.*`；相应 exit 均为 0。第一次接口 AST 辅助检查误把另一个 `per_sequence` comprehension 的 `mean_iou` 当作 tracker 字段，退出 1；这是审查辅助脚本错误，修正作用域后通过。原脚本及错误输出完整保留于 `interface_checks_attempt1.py`、`interface_attempt1.*`。前后证据见 `snapshot_before.json`、`snapshot_after.json` 和两份 manifest 副本。

## 仍待履行的既定义义务

1. R2 原任务释放授权 GPU 后，完成真实 R3 视频/crop、干预/replay、证据收集、有限梯度、candidate-only 更新、parent 保持和 strict reload 的资格验证；本次不查询或新增 R2 门禁。已有 synthetic CPU 合约不能替代它。
2. 实现已声明待做的完整训练总控和资格程序，运行规定的 raw/calibration/fresh-consequence 训练并收集真实 training/controller 退出。缺少该总控不计作这八个入口新增 bug。
3. 实跑六次 full98 与同一 positive checkpoint/config 的三套全量官方评测，保留初始化和负结果，再验证各模块独立作用、匹配控制/成本和全部七项 baseline+2。当前训练、正式指标、效果目标均未完成；不增加三 seed 要求。
