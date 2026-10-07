# Identity evidence transfer 源码评审

2026-10-07 13:48:32 CST。结论：**PASS**，scope 为 `COMPLETE_IDENTITY_EVIDENCE_PIPELINE_SOURCE`。当前没有未解决的 BLOCKING 或 NONBLOCKING 源码问题。

评审者为 fresh agent `/root/identity_evidence_transfer_review`。按本地策略请求 `gpt-6-astra` / `max`；实际 serving backend 和 reasoning effort 无法独立验证，因此记录为 `INDEPENDENTLY_UNATTESTED`。这是同模型家族评审，`review_independence=same-family`、`acceptance_status=provisional`。

直接读取了 `source_preparation.json` 所列七个源码文件、实验计划末节，以及候选提取、GOLA token 顺序、原 ABC 状态和选择器、控制器环境及指标收集依赖。没有编辑实验实现，没有运行 SSH、GPU、推理或训练。

已确认正确的部分：

- 严格加载完整 ABC parent 后再加入已训练的 encoded epoch 9 投影；实际生产者保存格式与加载器一致。局部已完成记录为 encoded_lr4、32 epochs、896 optimizer steps。
- GOLA normalized token 布局与取片位置一致。候选投影形状为 `B×7×5×2×128`，固定初始语义锚为 `B×2×128`，得到 `B×7×5` 支持分数。FP16 缓存往返后 FP32 投影与前序训练一致。
- 语义锚只由首帧归一化模板的 self-context 初始化一次；后续 query-conditioned template tokens 不替换它。原 A 记忆与 B 的 pre-attention 运动描述符保持原更新路径。
- 新残差相对原 C1 keep candidate 计算，对 regular/pause 两个动作等量相加；原质量、搜索和写入证据保持原值。零系数跳过分数运算。未执行区域继续由既有 valid/available mask 排除，每帧最多一个额外视觉前向。
- 在线只读取首条初始化标注，后续 GT 由离线收集器评分。四项真实视频 sanity 在完整运行前检查零系数父策略一致性、固定锚、梯度冻结、残差公式及 GT/未观察编码输入不变性。
- 四个固定系数 `0/.05/.1/.2` 各运行完整 98 序列 / 49418 帧；零系数预测文件逐字节、原决策数组逐项对照父策略归档。FULL98 收集器的五组参考、实际 GT 和 `sequence_mean_iou` 字段与控制器一致。
- 仅在四个 DEV arms 中选择一个系数，精确并列保留零系数，随后两套 native 数据集传入相同 parent/projection/weight。完整 shard union、LasHeR 245/220703、RGBT234 234/116649、五项 native 指标、31 个属性设置和 paired sequence intervals 均接入现有正确的输出格式。
- 运行规模固定，有 20 GiB 磁盘及 GPU memory/utilization 前置检查；每波所有子进程均等待结束并关闭日志，再检查退出状态，失败阻止后续阶段。

初读时的一项 NONBLOCKING 记录覆盖问题已修复：原 `all_states_exact` 声明缺少 branch mask、committed tensors、pending/score/box deques 和 history 长度的直接比较。执行者已补充这些既有状态以及 frame、branch metadata、各原始 anchor 的比较，并改为 `zero_residual_all_parent_states_exact`。本评审已重新读取实际修订文件，确认断言与当前 `Branch` 数据结构一致。

最终七文件独立 AST 检查通过，零模型导入。**真实 GPU sanity、FULL98 和 native 运行仍为 PENDING**；本记录不构成任何运行时 PASS 或精度结论。必须按已实现的顺序先通过 sanity，再完成 FULL98 与零对照，再锁定策略并完成 native sanity/full runs。

本阶段新增优化器更新为 **0**。前序投影的 32 epochs / 896 updates 只属于 identity pretraining，不能算作新的完整 ABC 训练。当前五项指标均加 2 的主目标仍为 **ACTIVE_UNMET**。
