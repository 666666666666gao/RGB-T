# 实验跟踪（2026-10-02）

| 任务 | 状态 | 证据/备注 |
|---|---|---|
| 官方源码与完整权重 | DONE | upstream commit 339c737cda6a24be667b6e5abdc721e8d6046f05；gola_b224.bin 与 DINOv2 主干缓存 |
| 环境 GPU smoke | PASS | /data/gb/setup/smoke_gola.json；随机权重，仅环境验收 |
| 官方作者结果校准 | DONE | LasHeR 77.5/73.9/61.6，RGBT234 92.2/69.5；来自作者预测，不是新推理 |
| 原 pretrained eval | FAILED | 18:47:38 Gloo 默认 30 分钟超时，未进入推理 |
| 原基线 10 epoch 训练 | STOPPED | 20:31:54 按用户新优先级停止；20:32:37 确认无进程，尚未 GPU minibatch |
| LasHeR train 索引 | DONE | 979/979 完成；MMOT .np 与 Video .pkl 保留 |
| C1 代码审查 | PASS | same-family provisional；无BLOCKING；val默认值已与计划对齐 |
| C1-sanity | PASS | 20:59:16启动；8steps×4；新参数117889更新、base无梯度、有限loss；8验证样本无失败，不证明恢复 |
| C1-initial | DONE | 21:01启动，21:08左右完成；12288训练采样；256val IoU .744937→.750184，4/8重新选对，2/227正确样本变错 |
| C2/C3/A/B | NOT IMPLEMENTED | 固定分支、状态恢复、rollout、判别压缩、多未来尚待实现 |
| 官方测试与主结果 | PENDING | LasHeR test、RGBT234；VTUAV deferred |
| GitHub上传 | DONE INITIAL PUSH | 补全upstream历史后d6f1544成功push；后续完整结果/文档待追加提交 |
| Online代码审查 | PASS | independent same-family provisional；修正训练/推理dtype、attention、计时记录 |
| Online C1/baseline smoke | RUNNING | 21:31:05开始，LasHeR/RGBT234各2序列×64frames，截断仅功能检查 |
| Online核心数据集完整评测 | PENDING | 独立入口避免test元数据索引；四任务，随后官方指标 |
