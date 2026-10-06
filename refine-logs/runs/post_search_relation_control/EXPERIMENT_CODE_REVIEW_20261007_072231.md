PASS — 部署前源码审查，当前无未解决的 BLOCKING 问题。

审查时间：2026-10-07T07:21:57.114445+08:00。范围：`COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE`；运行范围：`DEPLOYED_POST_SEARCH_BIDIRECTIONAL_CONTROL`。

这是独立上下文、同模型家族的暂定审查（`same-family` / `provisional`）。请求的 reviewer 配置为 `gpt-6-astra` / `max`，实际后端模型与 reasoning effort 未独立认证。`runtime_verified=false`；未执行 NN、训练、SSH 或部署，未宣称跨家族通过。

已关闭的两处具体阻塞：

- `research/check_post_search_relations.py:53` 原先要求无 extra 时整个 `post_search_scores` 为零，但这些辅助值仍含 action MLP、risk 与额外区域成本。当前第 53–55 行改为核对 `observed_output(region=0)` 不改变输出，并保留原本地分数相等检查。已复读修正源码。
- `scripts/run_post_search_relation_control.py:46`、`:49` 的初始分片合并原先写 `smoke_only=False`，与 `research/collect_recoverability_metrics.py:196–198` 的带 split 开发集约定冲突，必定阻塞后续 GT 报告。现在两处均为 `True`，同时保持完整 98 序列、49,418 帧、`max_frames=0` 和完整分片联合校验。这里的历史字段名不表示截帧。

计划元数据也已修正为 `epoch0_full98_required=true`、`epoch0_full98_measured=false`，没有把未运行项目标为已测量。

最终审查的是更新后的九候选计划：四组同初始化、同数据、同七上下文目标的训练，LR 为 1e-5 / 3e-5，seed 42、batch 336、各 64 epoch / 576 更新。八个 best/last 加一份共享的 bidirectional epoch0 完整 98 序列结果，在看 TEST 结果前按开发集 sequence mean IoU 选定同一个权重。epoch0 参照由 25/25/24/24 个完整序列分片合并；若它被选中，`selected_epoch=0`，不能记为新的学习收益。该选择规则已在运行前写入计划。

核对结果：

- `candidate_relations.py:39–52` 的 pair 输出为 B×6×2×5×7。各 extra 只和 local 同处一个 attention 上下文，没有 extra-extra 可见性；没有新增参数。索引写入保持可微路径，score 的原地调整不改写 backward 所需的 sigmoid/tanh 输出。
- `recoverability_modules.py:190–214` 同时重算 local 和 extra，以及 keep 的 advantage、harm 和 risk 参考；`:233–271` 在应用 pair 输出前完成搜索决策。`recoverability_tracker.py:168–181` 先执行本地决策，再至多执行一个 extra visual forward，最终动作与诊断使用该对候选。
- `recoverability_modules.py:375–388` 对两组模型均使用七个等权上下文，candidate/action mask 限定为 local 加单一区域，budgeted ranking 限定当前区域，搜索监督仍使用原始 cache 标签。前向仅接收 `DECISION_FIELDS`，GT/future 标签仅进入训练目标与离线评分。
- `train_recoverability.py` 严格加载同一父模型，冻结 C1、记录并检查 A/B/C 与 relation 梯度和权重变化，保留 best/last strict reload 检查。新的 `initial.pth` 在第一个 optimizer step 前保存。旧 family 默认行为保留，评估与收集均按新 family 启用 post-search 行为并 strict reload。
- 控制器先要求前轮完整结束，再运行 causal check、四个真实两 epoch sanity，之后才启动四个 full fit。实际 full98 结果才进入选模。GT 报告核对实际数据标注；两个 native 数据集固定同一检查点，完整覆盖 245/234 序列、五项指标、19+12 属性、四个参照的 5,000 次 paired sequence bootstrap、曲线与机制记录。
- 所有消费者结束后才清理本轮 20 份权重中的 19 份；选中的一份保留。删除范围仅本轮 sanity/full 的 best、last、适用的 initial，父模型、GOLA、C1、old4、motion 不在删除目标内。
- 私有 deployer 与被动 observer 源码已检查。已有 source sync 检查保留；observer 首查 180 秒，此后 240 秒或按已完成 sanity 估计接近结束时间，排除 zombie，不启动重试或训练。

实际完成的验证仅限源码：8 个计划源文件、2 个私有脚本及其 2 段嵌入 payload 的 stdlib compile/AST 均 PASS；六个已修改的 tracked Python 文件 `git diff --check` PASS。第一次 compile 命令命中了本地 PATH 的 `E:/Scripts/python.exe`，在读取脚本前报 `No pyvenv.cfg file`；同一检查脚本改用已有 `E:/python.exe -X utf8` 后通过，未安装或修改环境。这不是实验或 GPU 失败。

部署后的 causal/backward/timing、四组真实 sanity、full fit、九份完整98结果、两个 native 和清理回执仍需实际完成。源码 PASS 不证明显存可容纳、CUDA/autograd 已运行或方法有收益。开发98已被复用；固定权重的 paired sequence bootstrap 也不构成训练种子稳定性证据。

最终追加复核：observer 新增的两行仅把已存在的 `causal_pair_check.json` 读入观察结果；未增加查询、重试或 NN。已重新编译其 outer 与 embedded payload，PASS。实验源、deployer 与计划未因此改变。
