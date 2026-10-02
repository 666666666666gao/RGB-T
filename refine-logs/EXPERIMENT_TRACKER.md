# 实验跟踪（2026-10-02）

| 任务 | 状态 | 证据/备注 |
|---|---|---|
| 官方源码与完整权重 | DONE | upstream commit 339c737cda6a24be667b6e5abdc721e8d6046f05；gola_b224.bin 与 DINOv2 主干缓存 |
| 环境 GPU smoke | PASS | /data/gb/setup/smoke_gola.json；随机权重，仅环境验收 |
| 官方作者结果校准 | DONE | LasHeR 77.5/73.9/61.6，RGBT234 92.2/69.5；来自作者预测，不是新推理 |
| 原 pretrained eval | FAILED | 18:47:38 Gloo 默认 30 分钟超时，未进入推理 |
| 原基线 10 epoch 训练 | STOPPED | 20:31:54 按用户新优先级停止；20:32:37 确认无进程，尚未 GPU minibatch |
| LasHeR train 索引 | DONE | 979/979 完成；MMOT .np 与 Video .pkl 保留 |
| C1 代码审查 | PENDING | experiment-bridge 要求独立同家族 provisional 审查 |
| C1-sanity | PENDING | 尚未运行 |
| C1-initial | PENDING | sanity 后运行 |
| C2/C3/A/B | NOT IMPLEMENTED | 固定分支、状态恢复、rollout、判别压缩、多未来尚待实现 |
| 官方测试与主结果 | PENDING | LasHeR test、RGBT234；VTUAV deferred |
