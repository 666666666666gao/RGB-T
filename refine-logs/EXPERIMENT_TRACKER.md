# 实验跟踪（2026-10-02）

| 任务 | 状态 | 证据/备注 |
|---|---|---|
| 官方源码与完整权重 | DONE | upstream commit 339c737cda6a24be667b6e5abdc721e8d6046f05；gola_b224.bin 与 DINOv2 主干缓存 |
| 环境 GPU smoke | PASS | /data/gb/setup/smoke_gola.json；随机权重，仅环境验收 |
| 官方作者结果校准 | DONE | LasHeR 77.5/73.9/61.6，RGBT234 92.2/69.5；作者预测/包GT，不是新推理，RGBT234包GT与实际数据3文件不同 |
| 原 pretrained eval | FAILED | 18:47:38 Gloo 默认 30 分钟超时，未进入推理 |
| 原基线 10 epoch 训练 | STOPPED | 20:31:54 按用户新优先级停止；20:32:37 确认无进程，尚未 GPU minibatch |
| LasHeR train 索引 | DONE | 979/979 完成；MMOT .np 与 Video .pkl 保留 |
| C1 代码审查 | PASS | same-family provisional；无BLOCKING；val默认值已与计划对齐 |
| C1-sanity | PASS | 20:59:16启动；8steps×4；新参数117889更新、base无梯度、有限loss；8验证样本无失败，不证明恢复 |
| C1-initial | DONE | 21:01启动，21:08左右完成；12288训练采样；256val IoU .744937→.750184，4/8重新选对，2/227正确样本变错 |
| C2/C3/A/B | NOT IMPLEMENTED | 固定分支、状态恢复、rollout、判别压缩、多未来尚待实现 |
| 官方测试与主结果 | PARTIAL | RGBT234234seq/116649frames actualGT DONE；C1整体精度基本持平；LasHeR test在运行；完整方法未实现；VTUAV deferred |
| GitHub上传 | DONE INITIAL PUSH | 补全upstream历史后d6f1544成功push；后续完整结果/文档待追加提交 |
| Online代码审查 | PASS | independent same-family provisional；修正训练/推理dtype、attention、计时记录 |
| Online C1/baseline smoke | PASS | 四个128帧检查；首框/数量/有限值/延迟核验PASS，截断仅功能检查 |
| Online核心数据集完整评测 | RGBT234 DONE / LasHeR RUNNING | 22:35 LasHeR两原session活跃90/245、73440帧，ETA约105分钟；RGBT234两组234/234、116649帧全量完成，actualGT来源已验证 |
| RGBT234真实GT来源修正 | PASS / DEPLOYED | fresh reviewPASS；原生loader逐序列/模态及图像数PASS；只CPU重评分、保留旧包GT日志、无推理重跑 |
| RGBT234正式C1初轮效果 | INCONCLUSIVE / ESSENTIALLY TIED | baseline MPR91.778858/MSR69.232593；C1 MPR91.817009/MSR69.233890，差+.038151/+.001297百分点；继续C2/C3前须分析收益/损害，不主张显著提升 |

## 2026-10-03 integrated ABC campaign (current)
| Stage | Status | Evidence / next gate |
|---|---|---|
| Original fullGOLA-B/C1 two-dataset evaluation | COMPLETE | e72b9bb full native/all diagnostics/5CI include0 |
| New ABC acceptance | ACTIVE_UNMET | all5 overall standard metrics >=baseline+2pp, no narrowed claim |
| A/B/C + causal collector/train/online interfaces | SOURCE_PASS | abc_temporal_review final PASS for M0; whole-prefix256 causal training |
| Complete-prefix memory training correction | IMPLEMENTED_PREDEPLOY | reviewer actual24frame mismatch corrected to256>=maxprefix |
| Real new ABC sanity | PASS_NO_GAIN_CLAIM | actual8TRAIN/8val;2optimizersteps;ABCchanged/C1frozen;2x128 onlineframes;0switches |
| Collection / joint training | M1_RUNNING_M2_PENDING | three free GPUs1/2/3, batch32;3072TRAIN/512val planned;GPU0otheraccount occupied |
| Full ABC official benchmark | NOT_RUN | after internalvalidation, all245/234 andcomplete native/branch-aware metrics |
