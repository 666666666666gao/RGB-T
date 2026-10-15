对封存版 R3 consequence 新增源码的独立审查，2026-10-15。

**结论：FAIL_SOURCE_METADATA；blocking_count = 1。** 新训练入口存在一个确定的 checkpoint 元数据缺陷，应修复后再部署该入口。未发现其他确定的采集、因果或损失计算阻断。此结论仅针对原封存版，不包含执行方随后修改的工作树，也不影响正在运行的 R2 H/HA 20 epoch 作业。

请求 reviewer 为 Astra / max；实际 model、route、effort 均为 **UNATTESTED**，没有 host attestation。`review_independence: same-family`；`acceptance_status: provisional`。这是 fresh source review，不是实际 Torch/GPU 资格验证或完整 A/B/C 方法验收。

审查输入为 `C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/dense_consequence_source_review_inputs_20261015/` 的 17 个 manifest 文件。完整审查新增 `dense_consequences.py`、`collect_dense_consequences.py`、`train_dense_consequences.py`，并沿封存依赖核对其真实调用路径；其余既有 R3 接口以 `PREVIOUS_SOURCE_REVIEW.md` 为范围背景，未为增加检查数重复审查全部旧实现。下列源码路径相对封存的 `project/cadtrack_abc/`。

**B1 — P2，部署新增训练入口前必须修复：零更新候选错误宣称决策头已经训练。**

`train_dense_consequences.py:114` 在构建共享 config 时无条件写入 `future_actions_trained=True` 和 `search_actions_trained=True`。`:133` 把 updates 初始化为 0，`:142` 在进入任何训练循环之前调用 `save(0)`；`:139` 将上述 config 原样写入 `candidate_epoch00.pth`。因此，当声明的 dense-motion 父模型两项标志为 False 时，epoch 0 的权重仍是父模型原始权重、更新数是 0，却被标为两头已训练。共享 `manifest.json` 也在更新前携带这些 True 值。该错误会出现在正常入口，不依赖罕见输入；保留 epoch 0 作为候选使错误产物成为实际选模输入。`full_ABC_training_complete=False` 不会纠正两项具体标志的错误。

私有的 `check_epoch0_metadata.py` 只提取该文件中 config 赋值、save 函数及 save(0) 三段 AST，用标准库替身接收保存内容。以父标志 False、updates=0 推导出的保存标志确实均为 True，实际退出码 0。这是元数据控制流见证；没有导入 Torch、项目代码，也没有保存或执行真实权重。

最小 patch 建议如下。保留父模型已有标志，在每次保存时分别结合本轮实际更新数；这样已训练父模型继续训练的 epoch 0 仍能如实为 True，无须兼容 fallback 或额外门禁。

```diff
-    config = dict(parent['config'], stage='consequence_policy', future_actions_trained=True,
-        search_actions_trained=True, consequence_parent=str(parent_path), consequence_parent_sha256=parent_sha,
+    config = dict(parent['config'], stage='consequence_policy',
+        consequence_parent=str(parent_path), consequence_parent_sha256=parent_sha,
         consequence_epochs=epochs, consequence_batch=args.batch, consequence_lr=args.lr,
         consequence_seed=42, consequence_probe=args.probe)
@@
     def save(epoch):
         path = output / f'candidate_epoch{epoch:02d}.pth'
         weights = dict(parent['dense_abc'], decisions=model.state_dict())
+        checkpoint_config = dict(config,
+            future_actions_trained=parent['config']['future_actions_trained'] or updates > 0,
+            search_actions_trained=parent['config']['search_actions_trained'] or updates > 0)
         torch.save(dict(model_family=parent['model_family'], stage='consequence_policy',
-            dense_abc=weights, epoch=epoch, updates=updates, config=config), path)
+            dense_abc=weights, epoch=epoch, updates=updates, config=checkpoint_config), path)
```

终态两项 True 位于实际更新和两组参数变化检查之后，可以保留；probe 的 True 只代表确有有限更新，不能被解释为完成正式训练。修复后只需对该改动做一次有界复审。原封存源码未修改；本报告不将执行方的修复事实反写为原版通过。

其余重点路径的源码结论如下。

- **同状态干预、实际前缀和时间顺序一致。** `dense_consequences.py:52` 在观察前保存 prior，`:55` 分别从 local 构造三个 extra，`:68` 与 `:71` 对比重建动作和真实 track 的输出及状态；每个干预在 `:84` 恢复 prior，当前 advance 后才跟踪 future_images。`:97` 恢复的是真实当前帧 native_state，采集器下一查询从该实际前缀继续推进。`collect_dense_consequences.py:57` 仅用 frame 0 GT 初始化；`:64` 按全部中间帧推进，不跳到 GT 中心或短 clip 初始化。未来图像仅用于标签 rollout；GT 进入有效性检查和 `label_rollout`，不进入候选、动作、搜索特征或在线提交状态。
- **合法上下文、动作顺序和搜索目标相符。** 三个 extra 按 motion 概率稳定降序排列（`dense_consequences.py:54`），与运行时 `dense_main_tracker.py:95` 的 rank-to-mode 映射一致。`observe_extra_patch` 要求未扩展的 local query（`dense_main_model.py:96`），每个上下文是 5 或 10 个候选，三个 extra 不互相混入。候选外层、六动作内层的采集顺序与 `[B,N,6]` flatten 索引一致，总数 `(5+10+10+10)*6=210`。`train_dense_consequences.py:60` 用当前 future selector 的实际 selected_action gather 每个上下文的 utility，再训练 search（`:66`）；没有用 oracle 最大真实 utility 替代当前 selector。
- **R3 提交语义保持。** 新采集器仅复用 R1 的动作名称与标签函数；干预调用的是原生 R3 `tracker.advance`，没有调用旧 `advance_candidate`。`dense_submission.py:74`、`:79` 及 `dense_main_model.py:129` 使 defer_geometry 同时拒绝几何历史和所有模态记忆/模板写入；pause_identity、accept_geometry、RGB-only、TIR-only 的掩码仍按已审接口执行。时间继续推进，受保护的 anchor 和未获许可模态保留。
- **标签类型、坐标、形状和训练损失协调。** 原生 trace box 为 xywh，适配器变为 CPU float64 `[1,4]`，两个模态提交标志为 bool `[1,2]`，几何提交为 bool `[1]`（`dense_consequences.py:27`）。`counterfactual.py:152` 拼接后对预测和 GT 一致转 xyxy，再用有效 GT 计算 IoU；当前帧有效且未来至少一帧有效的条件也由采集入口先检查。缓存 utility 等为 `[N,6]`、future_iou 为 `[N,6,H]`。action features 为 `128+128+5+2+6+1+4+1=275`，search summary 为 `128+128+36+3=295`，批处理后的形状与各 head、loss 对齐。未来 harm 单独取未来 IoU 下降或 lost fraction 上升，不是总 utility 负号。以上是源码推导，没有执行真实张量。
- **训练范围、划分和产物语义基本明确。** 训练器只实例化 DenseDecisionHeads，optimizer 只含 future/search 参数；其余父组件直接保留在保存字典中。train/validation receipt 要求同一父 checkpoint，序列集合必须非空且 disjoint（`train_dense_consequences.py:102`）。缓存验证被明确标为 diagnostic，不执行选优或输出正式指标。默认 20 epoch 保存 0/1/2/5/10/20，其他终止 epoch 另行保存；probe 最多两次更新。最终 reload 对 decision state 使用 strict=True 并逐项比较；完整 tracker 的 strict 装载仍需后续真实运行确认。B1 是这些产物中已确认的标志错误。

仍需履行的既定运行与后续实现义务：

1. 在既有 R2 完成和 R3 运行准入安排下，实际执行小规模原生采集，确认 replay/snapshot/restore、强制 extra、几何隔离和模态提交在真实图像与模板更新时成立；运行 feature/label 的实际张量路径。这次静态审查不能替代该验证，也没有授权提前查询或触碰当前 R2。
2. 用实际收集的互斥 train/validation 数据完成有限训练资格验证：有限 loss/梯度、两组参数实际变化、父组件保持、保存后的 decision 及完整 tracker strict reload。按修复后的代码检查零更新候选与 probe 状态。这些是现有阶段的真实运行义务，不增加 hash 门禁、环境安装或三种子要求。
3. 补齐已明确声明的 local/one-extra 候选证据校准，以及完整视频选模和正式评测接线；这些缺失属于后续工作，不另报为本次声明入口的源码缺陷。有限 rollout 监督描述的是冻结教师后续策略；缓存收益不能证明更新后策略在连续视频上的收益。
4. 完成 full98 连续视频选模，并以同一选定 checkpoint/config 评估 LasHeR、RGBT234、RGBT210 全部七项标准指标。全 A/B/C、全部 baseline+2 目标仍未达成，本报告不作训练完成或性能承诺。

执行证据仅有少量标准库检查：17 个 manifest 记录匹配现有字节/摘要、13 个 Python 文件 AST 解析及 JSON 解析，以及 B1 元数据 AST 见证。初始检查与见证真实退出码均为 0；脚本、stdout、stderr 和 exit 文件保存在 `private/`。最终完整性检查真实退出码 0，确认 17 个封存输入和 input_manifest.json 自初始记录以来字节未变；见 `private/final.stdout.txt`、`private/final.stderr.txt`、`private/final.exit.txt`。没有 Torch、NumPy、cv2 或项目 import，没有 SSH/network/GPU/训练/进程日志/observer/资源监控/安装/Git 修改，也没有新增 agent。未修改封存输入或工作树。
