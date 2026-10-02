# 当前实验计划（2026-10-02）

用户优先级：尽快加载完整预训练权重，先训练新模块观察效果；只做 LasHeR 与 RGBT234，VTUAV 暂缓，不做扩展数据集。完整论文目标为 A 判别保真有界历史、B 身份条件多未来、C 未选中候选与内部状态纠错。本计划遵循用户原方案的落地顺序，先 C 再 A/B，不把 C1 原型等同于完整论文方法。

## 当前部署：C1 质量与排序监督

- 冻结 DINOv2 ViT-B/14 + 完整 `gola_b224.bin`，通过官方 LoRA 合并加载。
- 独立入口 `python -m research.train_candidate`，只直接读取已经建立的 LasHeR train MMOT 缓存，不建立 test 索引。
- 首帧身份模板；较早有效观测在线模板；当前搜索图。历史间隔最多 10 个有效观测。首帧无有效框的序列不进入此阶段采样。
- 30% 在线模板使用平移扰动，模拟污染；这是合成扰动，尚不是模型自身失败轨迹。
- 候选：保留 GOLA 原赢家，补充空间不同的局部峰；最多 5 个，网格距离至少 2、框 IoU < 0.7。不足时用 mask，禁止把重复候选算作多个假设。
- 新头：768 维冻结融合特征投影到 128，附加 10 维证据（原分数、Hann 后分数、两个传感器各自对初始/在线模板的 patch 相似度、框坐标），预测原置信度 logit 的残差。零初始化末层，初始选择保持 GOLA 的 0.45 Hann 规则。
- 监督为真实训练框与候选框的连续 IoU；BCE 质量损失 + IoU 差超过 0.1 的候选对排序损失，权重 1。未选中候选不是自动负样本；不声称完整干扰物身份标签。
- seed 42，LasHeR train 序列随机划出 10% 开发验证；官方 test、RGBT234 不用于训练、调阈值或选 checkpoint。
- AdamW，lr 1e-4，weight decay 1e-4，batch 8，workers 4，bf16。新参数更新、base 无梯度、有限损失是 sanity 的必要条件。

| Run | 优先级 | 配置 | 完成条件 | 当前状态 |
|---|---|---|---|---|
| C1-sanity | MUST | 1 epoch × 8 steps；val 2 steps；batch 4，workers 2 | 真实数据训练、权重正确加载、仅新参数变化、JSON/CSV/checkpoint 生成 | PASS |
| C1-initial | MUST | 3 epochs × 512 steps；val 32 steps；batch 8 | 观察 held-out 候选质量、恢复机会、错误切换；结果可为负 | COMPLETED；详见唯一HANDOFF |

C1实际initial约7分钟，原0.5–2小时估计偏长。每180–300秒或接近实测预计结束时检查，不反复扫描数据目录。

## 后续完整实现（未部署）

1. C2：固定分支数与窗口，短期候选路径、跨帧累计证据、锚点保护，切换同步恢复在线模板和运动状态。严格在线，过去输出不改写。
2. C3：真实模型失败历史、不同候选短期 rollout 的后续收益监督。训练未来真值只用于标签，测试不能读取未来。
3. A：可信共享有界记忆和分支暂存，候选判别差距保持损失，对照等容量 FIFO/EMA；已不可逆长期压缩不宣称无限精确撤销。
4. B：身份分支条件的多未来运动分布，软约束与无强运动限制的视觉路径，等候选/搜索预算比较。
5. LasHeR test 与 RGBT234 严格在线评测完整 pretrained GOLA 和增强模型，使用官方 PR/NPR/SR、MPR/MSR，包含速度与高分位延迟。C1 crop IoU 指标不等于正式跟踪结果。

主结果未完成前不启动扩展基线、消融网格或扩展数据集。

## 当前下一步：C1 的严格在线效果验证

C1初始3epochs已完成；256固定开发验证crop上IoU .744937→.750184，8可重新选对机会中4成功，同时227原本正确样本中2变错。该小样本正向变化支持接入在线检查，不构成正式跟踪增益证明。

新增 `research/evaluate_online.py`，复用官方裁剪函数、SimpleProvider（area4/minsize10）、SimpleTemplateUpdater（.84/area2）、postprocessor（Hann .45），完整预训练模型、首次GT初始化。只读取第一行标注，当前预测驱动下一帧裁剪与在线模板；过去输出不改写。与旧eager框架相比直接按目录读图，避免重建全部图像header索引。C1改变候选选择，保留选中候选的原GOLA置信度作模板更新门控，不引入C2状态修复。

- Online smoke：LasHeR test与RGBT234各首2序列、每序列前64帧，分别baseline/C1；检查输出有限、帧数、首框、文件格式、真实模板更新，**截断结果只用于功能检查**。
- 完整评测：4个独立任务，LasHeR baseline/C1、RGBT234 baseline/C1，分别使用GPU0/1/2/3。只做现有两核心数据集。estimated 1–3 hours，取得在线smoke吞吐后修正；不把并行HDD成本忽略。
- 每个结果目录分开，保存每帧输出与包含读图/裁剪/推理/模板更新的延迟，最终调用 `evaluation.py` 的原rgbt指标函数。RGBT234需显式读取实际数据标注：包GT与实际3文件不同，已修正来源，公式未变。
- 全部推理默认float16，匹配原GOLA inference配置；C1训练bf16的差异写入config；对照两种方法推理dtype相同。
- 部署前对此新增online入口做独立代码审查；先完整smoke通过再启动4个完整任务。C1仍不代表完整A/B/C2/C3。

## 当前C2原型功能验证（23:00准备）

RGBT234实际GT全量C1结果基本持平（MPR+.038151/MSR+.001297百分点），LasHeR仍在原进程运行；不依据官方test选新参数。继续原方案的有限分支/状态修复实现，先在LasHeR train-held-out验证。

- 代码 `research/bounded_recovery.py`、`research/probe_recovery.py`。完整pretrained和C1 epoch3冻结；本阶段没有新增可训练参数，也不声称C3后续收益监督已经完成。
- P=3，W=5（含出生帧），每分支独立search缓存/在线template/mask、近期boxes/scores、待提交模板；首帧锚点永不覆盖。新fork从父分支已提交模板起步，不继承当帧赢家的新更新。
- 所有分支仅消费当前已到帧；C1质量分数5帧均值作临时路径价值，挑战分支持续2帧高于主分支才切换。切换使用该分支自身search/template/pending；未选分支窗口到期丢弃，旧主分支降级时重新获得5帧窗口。当前无B运动预测，保留的box历史服务后续B/C3。
- matched box-only对照使用相同选择规则，但切换时沿用原主分支template/mask/pending；因此后续路径可以分歧，不能宣称两者每个切换动作完全相同。
- sanity固定内部validation排序前2序列 `10rightblackboy`、`1boygo`，每序列前128帧，顺序运行C1/C2/box-only；计时受缓存顺序影响不作加速结论。GT仅首框输入tracker，其余外层事后诊断。
- 验收：真实前向、输出形状/有限值、anchor不变、分支/各deque不越界、按有效GT排除初始化的IoU；记录实际switch/commit。若无switch，不宣称已用真实运行验证状态切换。保存当前主分支和各分支框，按同帧GT统计有益/有害切换；报告相对C1逐帧得益/损害及分母，不能把帧计数叫恢复事件。
- 先独立review通过，再GPU2 sanity；预计2–6分钟，180–300秒检查，失败读原日志，不自动改方法重跑。sanity后才决定是否扩至16内部序列512帧诊断、或优先接C3价值监督。两官方测试已运行的任务不修改/重启。

## 23:50 C3 教师/价值头首轮（内部开发，不能代替完整评测）

C2 的16序列内部诊断出现负结果；下一步按原C3机制监督未来价值。保持原LasHeR train内881/98分区，只用该分区生成预测历史；官方test/RGBT234不输入训练或选权重。首轮sanity train/val各2clip，history1..8、H2、batch-clips2；通过后初始train128/val64，history1..256、H3、batch-clips8（最多40未来动作共享主干batch）。冻结完整GOLA和C1，未来图像/GT仅用于同一固定C1延续策略的训练标签，查询时输入在读取未来前复制保存。标签 mean future IoU-.1*wrong-write fraction，wrong-write为raw>.84且GTIoU<.2，仅定位污染代理。C3头接768features+10evidence+9history/motion，C1 projection初始化，训练batch32、20epochs、AdamW1e-4/wd1e-4、seed42，按内部val mean regret选择；记录改善与恶化clips、MSE、oracle/regret、真实参数变化/显存。缓存样本数小，首轮仅功能/学习诊断，不宣称完整训练或论文增益。真实batch32前后向容量测试峰2975.94MiB，对比batch8峰1015.98MiB；单次含warmup，不据此声称加速。

新增完整评测要求：只做LasHeR245seq/220703frames和RGBT234234seq/116649frames；两种完整baseline/C1都要PR/NPR/SR或MPR/MSR、所有属性/序列、全曲线、p50/p90/p95/p99/FPS、全失败/恢复事件及paired收益/损害，输出JSON/CSV；候选Recall/误拒恢复/更新污染代理需补时间线，不能用缺失值假充完成；尚未实现A/B不产生虚构预算/校准曲线。

## 未训练C1完整轨迹归因检查

实验源目前baseline postprocessor乘224为float32，C1 candidate boxes先转double再乘224；定位头明确reg_mlp输出转float32后sigmoid，确实是不同的浮点计算路径。未训练C1残差层初始化全0，原计划仅证明Hann赢家保持；需要确认整个在线轨迹/模板更新是否等价。对原LasHeR train-heldout按排序前6个序列做完整长度控制（offset0/3各3seq，GPU2/3在RGBT完整candidate工作完成后可用）。相同完整pretrained GOLA、相同firstGT、相同float16和搜索/更新规则，分别headNone与零残差新头；后续GT不读取，不算PR/SR。存所有rawXYWH、3decimal轨迹差、最大差、模板更新次数、改选次数，报告是否为精确初始控制，不把这6视频当官方数据集分数。若实际不等价，先展示代码与真实证据再修复，不擅自把已公布C1成绩全部归于学习。

### 01:50 observed numerical mismatch and corrected full protocol
Six fullTRAIN-heldout zerohead controls completed1401frames: no alternative choices andidenticalupdatecounts, but1boygo max2.004584px/33roundedframes; total45roundedframesdifferent. This is actual autoregressive amplification, not hypothetical. Minimal C1 inference decode now multipliesfloat32 by224 beforedouble, identical originalpostprocessor; weights/features/Hann/update unchanged. Preservev1 fullpredictions/reports and zeroheadv1 receipts. FreshreviewPASS correction, repeat zerohead NEWv2dirs before formaldeploy. After rawfull6 equalityverified, NEWfullv2C1 evaluations of bothcorebenchmarks onGPU2/3 with simultaneous candidate recording; never overwritev1. Same-run candidate references are explicitly labeled as consistency, not independent rerun equivalence; fullcoverage/actualGT/selectedtimeline checks stillrequired. Baselinefullv1 unchanged. No test-based hyperparameter selection.

### 02:49 C3 expanded TRAIN-only continuation after full RGB completion
ActualoldLasHer candidate audits finished02:22/02:23 andv2RGB fullfinished02:40; freed GPUs0/1 available. KeepC1officialruns/checkpoint untouched. C2andC3teacher stillusedold double-beforemultiply (bounded_recovery.py:93 andcollect_rollouts.py:63), inconsistentwithcorrectedC1 decode evidencedbyrealv1zerohead2pxamplification. Minimal samefloat32multiplybeforedouble correction +teacherconfigprovenance; freshreviewrequiredbeforedeployment. Preserve oldC2/C3source/artifacts.
After reviewed2clip TRAIN/val sanity, expand C3teacher512train/256internalval, original881/98 split, originalmaxhistory256/H3/lambda.1/seed42, batchclips32 (actualcapacity32/148actions previously9.286GiB), frozenfullpretrainedGOLA andC1epoch3. GPU0train/GPU1val; cachedutilityheadtrainingGPU3 onlyafterbothcompletion, batch32/20epochs/AdamWlr-wd1e-4 asoriginal, initializeC1projection. No test-derivedparameterchange. All20epochs/bestinternalregret/lastnegative/epoch0/causality/frozenbase/actualparamupdates/peakmem preserved. Stageisexpandedinternaltraining, notofficialC3PR/SR norfullABC. Review actual_source_advance confirms branch-local template updatedimmediately, soH3alreadyincludesitsfutureeffect; doNOT add hypothetical H>W fix ornewcommit logic.

### 03:13 corrected C2 internal repeat
ActualGPU3 free until expandedteacher completion; repeatoriginal16TRAIN-heldout/max512/P3W5patience2 withreviewedcorrectedgeometry, nonewparams/testtuning. Actualtmux c2_probe_corrected_v2 launched03:13:43; separateoutput/sourceSHA/oldnegativekept. Reportallvariants/pairedharmbenefit/switchstate/finiteboundchecks, notofficialbenchmark. Approx10min, checknear03:23.

### 03:56 full v2 and expanded C3 outcome (actual, not planned)
Both fullnative/core/candidate/actualGT reports complete245/220703and234/116649, finalizers/5000sharedpairedsequenceCI/complete_report merger markersactualPASS. All5CIinclude0. C1LasHeR PR77.860972726/NPR74.154517654/SR61.951313680; RGBT MPR91.817008897/MSR69.234047342. All19/12attrs/allseq/curves/failureevents/pairedharmrescue/calibration/updateproxies/latency/memory exported. DifferentparallelHDDload prohibitsfairaccelerationclaim. Modelparameters actualbase88093445frozen/C1new117889/total88211334.
Expandedteacher512TRAIN2407actions2495.115s10010.476MiB,256val1197actions946.121s9885.581MiB; allNPZshape/finite/choice/disjoint387vs91seqPASS. C3actual20epochs320steps119041paramschanged/base0,cachedfit1.240s83.781MiB. Bestepoch0 strictreloadall256exactmaxdiff0; epoch1ties0; trainingdoesnotbeatinitialization. Zeroresidualutility usespureC1qualitywithoutHann, soepoch0 +.003082672vsC1isinitialpolicychange, NOTlearnedgain. Last20utility.759239674belowC1.759344816,8better11worse35reselect,currentIoU.656989336; MSEdecreaseisnotselectionbenefit. Do NOTpromotebest0aslearnednewmodule. All21records/best/last/reload/plots retained.
Nextresearch work remainsA/B/onlineC2C3; no fullABCclaim. ActualexistingC3initialpolicyconfound notedinreviewandobservedbest0 meritscorrectingC3rankingto preserveoriginalC1Hann selection atzeroresidual, withfullTRAIN/valinitialchoiceequality andfreshcodereview beforedeployment. This correction isplanned, notimplementedorvalidatedatthisupdate; keep currentv2sources/artifacts unchanged untilreview. CurrentexistingtrainingandformalC1evalterminalSUCCESS; goalactiveuntilremainingauthorizedworkverified.
