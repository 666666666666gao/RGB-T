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

### 04:08 C3 v3 comparable initial-policy implementation
Actualv2bestepoch0purequalitydiffersC1Hann andlastutilitynotbetter; newminimalutility_selection_scores=evidenceHann+(predictedutility-rawscore)*.55 restoresC1atzeroutilityresidual. Regressiontargetsunchanged; rankingloss/evaluseactualsameHann-correctedscore. Mainassertswhole512TRAIN/256valinitialchoices==C1beforeoptimizer, saves explicitconfigequality/rank/windowfields. Freshgpt-6-astra/max reviewer c3_hann_control_review readingactualsourcebeforedeployment; no source deployedornewtrainingclaimedyet. Preservev2branch_utility/train_branch sources+allresults; short2epochsanityusingcompletedteacherNPZ, thenNEW20epoch320stepout withsameallhypers/teacher/labels, no testtuning. Evidenceforassertisobserved4/256initpolicyconfoundandreviewWARN, nofallback/compatibilityorunrelatedrefactor. FullC1v2officialscoresremainunchanged.

### 04:20 C3 v3 completed runtime outcome
Fresh fair-control reviewPASS; GPU2epoch sanity32steps then NEW20epochs320steps, changed119041params/base0, finitegrad/loss, cachedfit1.124049s83.784668MiB. ALL512TRAIN/256valzero-initialchoicesmatchC1,0mismatch;initialutility.759344816gain0reselect0. Bestepoch0:epochs1-5ties;laterallnegative,last20utility.759062052gain-.000282763regret.018815501,0better1worse1reselect,currentIoU.749436975,MSE.051513333. Strictreloadall256metricdiff0,768controlrerunPASS. All21epochs/rawconfig/receipt/curves/logs retained; oldv2sourcebackupandoutputs retained. Not officialonlineC3metrics; no learnedgainclaim. ONEhandoff includescurrentv3 + completeformalC1. RemainingonlineC2/C3andA/Boriginalresearchscope, allcurrenttrainingterminalSUCCESS.

## 2026-10-03 integrated ABC main method: all five native metrics +2 percentage points
User authorized implementation of ALL three modules and persistent improvement. Acceptance is SAME actual-GT/protocol full GOLA-B baseline: LasHeR PR>=78.598449/NPR>=75.101973/SR>=63.037425, RGBT234 MPR>=93.778858/MSR>=71.232593. Full245/234 sequences and all19/12 attributes, standard curves, sequence paired uncertainty, failure/recovery/harm, candidates/memory/calibration/cost remain mandatory. Optional user clarification on per-attribute +2 is pending; default five overall metrics, complete attributes including declines. Do not guarantee success or change scores/protocol to meet threshold.

Implement simple coupled learnable modules on frozen full pretrained GOLA plus frozen pretrained C1:
- A: separate RGB/TIR descriptors BEFORE mixed attention, protected first-frame anchor, four differentiable memory slots per modality, learned gated bounded compression. Train compression by retaining candidate identity margins from full available PAST trusted TRAIN history. Full history teacher used only for training targets; runtime memory fixed slots, branch-private pending and committed memory, no unbounded buffer.
- B: three-mode, three-step diagonal Gaussian motion head conditioned on protected identity + compressed memory and recent predicted boxes/time intervals. Causal history includes tracking mistakes, no GT-box decision inputs. Train future normalized centers/sizes from TRAIN GT labels; mixture NLL; standard deviations use softplus+.05 (positive lower bound, no upper bound); soft log-density evidence only, no hard visual exclusion.
- C: candidate current quality and future value supervised separately; actual v3 future-only regression did not improve selection, so retain current IoU supervision and learn calibrated future term. Unselected candidates use their quality labels, never auto-negative. Joint selector consumes A identity support/B soft motion and C1 features, with zero residual preserving original C1+Hann; counterfactual future labels use identical frozen C1 continuations. Bounded3 alternative paths, W5 reversible state, switch repairs search/template/memory/motion; original past outputs immutable.

M0 source review and real GPU sanity: collect small causal clips, actual optimizer gradients A/B/C, finite loss, actual frozen base, zero-residual policy, bounded private state.
M1 paired TRAIN-only temporal collection: frozen GOLA/C1 predicted prefixes batched across clips, current decision arrays saved before future images; all available past observations0..query-1 stored within history256>=maxprefix256, descriptors/evidence/current candidates and future GT/short rollout labels separately. Original881/98 split reused, no index rebuild/test sampling. Three free GPU collection processes: train1024 seeds42/43, then val512 seed100042 followed by train1024 seed44 on GPU3. Sampling with replacement:3072 sampled TRAIN clips/3032 pooled unique sequence-query pairs; original TRAIN/validation partitions disjoint. Initial collection estimate from actual existing batch32teacher memory9.8GiB, new batched prefix speed measured by sanity; reserve outputs<20GiB, available2.5TiB.
M2 cached joint training initializes C1 projections/head from bestepoch3, AdamW lr1e-4/wd1e-4,batch64,30epochs, internal fixedvalidation selected current-IoU with future-regret/harm diagnostics and last retained. Each module must receive finite nonzero gradients and change parameters; baseline parameters frozen. No official-test checkpoint/threshold selection. Expanded or revised training only if measured validation evidence warrants it.
M3 causal online ABC integration and internal complete-sequence control first, then fullLasHeR/RGBT234 using reviewed original metrics/GT and alldiagnostics. Four GPUs used for independent training/validation/control/full benchmark tasks when useful, never fake memory. Final complete eight ABC combinations and matched-cost controls only after main method works; all negative results retained.

Current2027 warmenv (/data/gb/envs/gola torch2.5.1cu118 Python3.10) reused unchanged; prior seeded GOLA/C1/float16forward and fresh doc-following witness already accepted. No environment rebuild/install. This plan is newly authorized, NOT completion evidence. Goal remains active until ALL modules integrated and +2 pp full-protocol evidence actually satisfies acceptance.

ABC revision before deployment: reviewer identified initial24frame truncation would lose full-past teacher and reset the student unlike persistent online memory. Collector history capacity now256>=maxprefix256 (explicit invariant), preserving every available frame0..query-1; A student replays the complete causal prefix and teacher uses its GT-quality trusted subset. B still consumes only recent8. Padding is masked and never writes memory. Full-prefix TRAIN-only arrays are offline training storage; online ABC remains4slots/modality +8motionboxes +P3W5 states. No truncated-history claim, no future input, no fallback. Initial256prefix collection estimate and training wall time will be measured by sanity.

2026-10-03 10:16 source+actualGPU complete ABC diagnostics/audit PASS. Fixed train_abc_after_collection.sh activated once (sleep240, no retry); val512 COMPLETE/all98val/503unique queries, train collection still live. Instrumented inference records allparents candidate sets, actual template source frames/boxes, real learned slot update coefficients/known-vs-wrong contribution mass, permodality forecast jointNLL/PIT/coverage/area. Native output convention --output RUN/predictions, allfull consumers --runs RUN.30epoch trainer metadata checkpoint_selection accurately states strictmax internalcurrentIoU; audit strictloads initial/best/last, retains all31epochs/calibration and pooled duplication. FullABC scores/+2 acceptance remain pending.

### 2026-10-03 11:28 actual M1/M2 completion and predeclared M3 selection
All three1024TRAIN collectors and512validation complete; 3072sampled/3032unique/40repeat TRAIN queries. Joint30epochs/1440steps/B64/seed42: A109569/B48843/C18691 changed, frozenC1exact/no gradients;707.236813s/9638.287109MiB. All31cached selectedIoU tiesC1 .749872207642;best0,0reselect/rescue/harm. Initial/best/last strictall512reloaddiff0 and all31epochs retained; calibration improved withoutchoicegain. Full98online TRAINvalidation49418frames each for initial/last/last_box_only running GPUs2/0/3; predeclare sequence-weighted valid-frameIoU for initial-vs-last selection BEFORE official test. Box-only is state-repair control, not checkpoint candidate. None of these are official PR/NPR/SR, no +2 claim; fullcore245/234 all31attrs/native/branch/forecast/cost/CI stillrequired. A new offline98collector is under limited same-family review; original algorithm frozen.
### 11:40 all98videos COMPLETE / frozenlast30 / full4GPU officialRUNNING
Allinitial/last/lastbox98seq49418frames49233valid complete. Predeclaredsequenceweighted fullonlineIoU initial .737194672342,last .739523904518,box .738778577820; selectlast30 at11:39:17 BEFORE officialABC test, tinygain+.232923pp vsinit NOTbaseline/+2. Detailed branch/slotforecast CPUcollector continues independently, compareaccuracywithselectionreceiptwhencomplete; do not waitCPUmechanisms to keepGPUuseful. run_abc_full.sh continuedreviewPASS noalgchange/fulloldbaseline receiptsverified. Fourofficial jobs launched11:40:05 GPUs0lasabc/1rgbabc/2lasbox/3rgbbox; full245/22070319attrs and234/11664912attrs/native5metrics/allseq/curves/events/branchslots/forecast/cost/5000pairedCI, fixedepoch30/P3W5patience2/seed42. RUN dirs /data/gb/outputs/abc_full_v1/{lasher_abc,rgbt234_abc,lasher_abc_box_only,rgbt234_abc_box_only}. FinalmarkeronlyallstagesPASS; sourcefrozen whilelive. NoformalABCscoreyet/all5+2UNMET. Fulltraining31epochPNG/PDFretained.
### 11:49 all98 mechanism report completed / negative uncertainty retained
CPUfull internal report finished; all3accuracyexactlockedselection. Initial/last/box failedframes8018/7487/7477,failureevents235/217/216,recovery222/235,206/217,205/216; switches514/446/463,benefit79/85/84,harm53/56/62. last-init+.232923ppCI[-.345903,.896431],23better13worse62ties. last-box+.074533ppCI[-.210859,.423822],8better10worse80ties. AllCIinclude0, fullstateNOTallmetricsbetter (RGB/TIRslotusedwrongfractions higherthanbox;217vs216events). Realtemplate/slot/write/forecast/calibration/provenance/all294sequence rows retained, nosemanticIDclaims. Fourfullofficialjobscontinue fixedepoch30; all5+2 UNMET. Add fullnativeABC-vs-boxmatchedreport afterbothfullsuccessmarkers, noextraGPU/inference/params.

### 16:35 revised user priority: budgeted completion and validated writing

Both formal core ABC/box-only/state reports and the actual total completion marker are complete; all five +2 targets remain unmet. Full original98 GOLA/C1 controls are also complete. The fixed three-frame post-correction pause prevents only one actual high-confidence write (a correct localization), so it does not establish safe-write efficacy. Full98 same-budget dense proposals preserve original GOLA/C1 weights and Hann winner: dense-Hann sequenceIoU .730286644 vs C1 .739440162 is negative; dense-raw .741248164 gives +.180800pp vs C1 but -.203034pp vs full GOLA, both paired intervals include zero. Expanded-pool C1 quality MSE .153/.159 vs original .0226 exposes a measured distribution mismatch; it is not sufficient to relax peaks and compare oracle recall.

Completed TRAIN visual audit has151 sampled missing-candidate queries/147 unique sequence-query pairs. Same causal batch contexts exactly reproduce original features/boxes/counts/choices. Original5 recall0; dense5 Hann10/raw17; widened original5+extra5 recalls25 but oldC1 cross-region selection0. All3 motions recall25 at four-region/up-to20 budget vs top-probability mode7 at two-region/up-to10 budget. These are GT-selected offline diagnostics, not deployable gains or independent trained seeds.

Next required data milestone is `research.collect_recoverability`: uniform TRAIN/held-out sequence/query sampling, frozen fullGOLA+C1 predicted prefixes to max1024, all prefix observations retained; current original5 plus six protected-anchor extra regions with up-to5 dense-raw candidates each. Extra proposals and original policy are executed before query GT. Add separate area-pooled RGB/TIR candidate descriptors without changing original C1 inputs. Learnable A/B/C inputs and GT labels remain separate. Save actual per-region candidates, current quality/presence and future3-step same-parent regular-write/query-paused-write action outcomes; pause leaves confidence, position and provider behavior unchanged and later ordinary updates remain enabled. Future images are first decoded after every query input is copied. One extra region is the eventual online budget; six alternatives executed here are teacher data and cost must be disclosed.

17:54 closure: this data milestone is COMPLETE, 768 sampled TRAIN rows / 767
unique sequence-query pairs from 519 sequences, 128 unique held-out queries
from 76 disjoint sequences. All arrays finite, complete causal prefixes and
regular/pause masks verified. Six-region oracle improvements are diagnostics.
Upgraded A/B/C source and fresh same-family CPU review passed; real M0 confirms
finite gradients and changed parameters for all three modules, frozen C1,
strict best-checkpoint reload, and zero-init C1 action parity. Batch64 H1024
capacity passed; batch128 also passed (4153.949MiB). Four independent
model seeds42-45 started17:57:42 on the same pooled data after capacity acceptance, batch128/30epochs; all had49 actual updates18:01. Keep only
validation-utility-best.pth for each independent run; preserve all epoch records.
Use one-extra-region budget and charge search even when no location switch.
The cache still uses frozen-C1 prefixes/continuations; collect the upgraded
method's own long/failure states later before claiming on-policy recovery.

Source review then realM0: TRAIN/validation two clips with maxprefix8; capacity M0 sixteen clips/maxprefix8, future forward microbatch64. Check frozen weights, ROI pooling, original C1 inputs unchanged, private state, finite arrays, action masks, every future label and actual GPU memory/time. Only after gates pass, launch train256 seeds42/43/44 plus held-out128 seed100042 on four available RTX3090. Default prefix batch16/future microbatch64/maxprefix1024, original881/98 split, outputs estimated under20GiB and >2TiB free. Use measured M0 progress to estimate completion; no official dataset tuning. This collection does not itself train upgraded modules or prove tracking improvement. New model training follows actual cache completion and source/M0 gates.

Checkpoint retention per explicit user instruction: after validation and audits finish, retain only each completed experiment's metric-best checkpoint. ABC retains pre-test full98-selected epoch30 (`last.pth`); C1 retains epoch3 `best.pth`; obsolete epoch/initial duplicates are removed, with metrics/logs/predictions retained. Temporary candidates needed for an outstanding checkpoint comparison are pruned once that comparison is complete. Never delete upstream pretrained weights or inputs of live jobs.

16:59 actual M0/launch update: source/CPU and realtrain2/val2 +capacity32 passed. Additional forward256 capacity32 completed38.234s/17100.568MiB with finite1297actions; limitedlauncherreviewPASS. Actualfull4launch16:53:37 uses prefixbatch32/forwardbatch256/maxprefix1024 (replaces prior planned16/64 after measuredcapacitygate). AllfourNNjobs confirmedLIVE16:59, actualGPUused~20GiB each;TRAIN first32 elapsed239–299s,validation64elapsed270s. Actuallongquery TRAIN197/768 andvalidation21/128. Estimatedvalidation17:03–17:10 andTRAIN17:25–17:40; nextcheck17:07. No optimizer/upgradedperformance claim.

18:12 complete: four independent seeds42-45 finished30epochs/180updates, retained only best7/6/4/6; ABCchanged/C1frozen/reload exact. Cache query IoU gain mean+.7858pp, utility+.7441pp; all descriptive paired sequenceutility CIs include0. Best pauses0 and missing recall gain0/1/1/1. Next: genuine online integration/source+GPU exact controls, thenfull98 matchedbudget/write/state evaluation, self-policy data collection and two complete official native reports. No formal performance acceptance yet.


## 2026-10-06T07:07:09.435730+08:00 selective-state-commit execution update

Actual four-GPU query M0 PASS, full1789labels collecting original3338360. Original3351735 queued: four2epoch fit sanities, fourfull60epoch C-extension fits (H3/H32 × lost0/.1, B128 LR.001 seed42), all98video selection, onehead forboth245/234 native/full5/31attrs/4pairedreferences. A/B+completeGOLA/C1/old4 remain active/frozen in this prototype; not jointABC retraining. Actual consecutive known-state events get totaltrainingweight1. Two27/32frame videos explicitlyH3-only; H32 doesnot inventfuture labels. OptimizerVAL exclusion, nofutureGTinput, no TEST checkpointchoice; onebestmodel sufficient, allfiveGOLA+2 stillUNMET. Humanhandoff only docs/HANDOFF_20261002.md; authoritative machine snapshot refine-logs/runs/selective_state_commit/current_execution_snapshot.json.


## 2026-10-06T07:45:41.497027+08:00 pre-fit endpoint-selection correction

Actual waiting-only3351735 retired with no fit/children/updates. Replacement3423188 waits unchanged3338360 collection; no NN collection/training replay. Four full60 fits preserve cached-best and actual60endpoint, strictreloadboth. All eight candidates each full98/49418 in two4GPU waves; selectONE by developer sequenceIoU before samehead bothnative/all5/31attrs/4refs5000CIs. Retire11 own unused heads onlyafter all consumers. Recipe/data/model unchanged, no>=3positive-seed gate, targetall5GOLA+2 remainsUNMET.


## 2026-10-06T10:20:55.846949+08:00 actual full-training/internal closure, native ongoing

Four C-extension fits each60/840 (not jointbackbone training), eightfull98/49418 complete; all learned checkpoints below frozenold4/gross. One epoch0 locked before bothnative; actual98prediction text equalparent, no training gain. TRAIN1646event1691queries/881seq and615write-pairqueries; threeeventroles overlap. Actual16CPUcachedforwards show TRAINpositive/VALnegative selectedutility despitepositive predictedutility. History244CSV/plots and report metadata copied; no NN/report replay. Currentoriginal3423188/native3643976-79 lastactual09:59:49, sole22853 next10:53:18 then240. Five formal new metrics pending; all5+2UNMET. One best sufficient, old4/dependencies protected.


## 2026-10-06T11:09:59.304261+08:00 actual full native closure; goal still unmet

Four60/840 fits, eightfull98 and SAME epoch0 both245/220703+234/116649 actual-GT complete. Five78.016094/74.383784/62.072660 and92.866361/70.022381, allbaseline+2false; exactgrossparent scores/no learned C benefit. Las threeCIpositive/RGBtwoinclude0; all31attrs/81metricrows/curves/four5000pairedrefs/mechanisms/efficiency copied3838files.11ownunusedweights5303694Bretired by original queue afterconsumers; bestepoch0 anddependenciesheld. Original3423188 andobserver22853 CLOSED; no replays or activeNN. Negative training and repeateddeveloper/test limitations retained. GoalACTIVE_UNMET, one best sufficient.


## 2026-10-06 current-frame and future-state reward alignment

The previous four60/840 fits and eightfull98 candidates are CLOSED negative;
their prelocked epoch0 native five metrics exactly reproduce old4/gross and all
five +2 targets remain unmet. Do not rerun the closed mean-all controls.

Actual completed-state CPU audit found207 H32 TRAIN queries with legal mode2
current-IoU gain>.03 and no worse future mean, all blocked by the .03 threshold
on mean-over-query-plus32futureframes;96 legal rescue actions were blocked.
The repeatedly used developer cache has only1 corresponding opportunity.
This is a TRAIN/developer oracle diagnostic, not demonstrated online efficacy.
Current+future oracle selection improves current quality but slightly reduces
old mean-all utility, so full videos must judge the tradeoff rather than cached
loss or the new objective alone.

Next four fits: H3/current_future/MLP, H32/current_future/MLP,
H3/current_future/linear, H32/current_future/linear. Current_future target is
current known IoU + mean of known future IoUs, with the existing optional lost
penalty applied to known future frames only; these four arms use penalty0.
Linear head is LayerNorm917 + Linear917-to3 (4588parameters), versus the original
119725parameter MLP. Both preserve exact zero-head parent behavior. This tests
the measured reward dilution and observed TRAIN-positive/VAL-negative fit with
two explicit axes, using the SAME completed1789 labels/event weighting, seed42,
batch128/lr.001/full60/840updates per arm. Full GOLA/C1/old4 A/B/C and frozen
motion remain active and frozen; this is C-extension training, not backbone or
joint A/B/C retraining. No future GT enters decision features.

Source review then real four-GPU2epoch backward/strict-reload sanity; full fits
only after all four pass. Preserve cached best and actual60 endpoint for each;
all eight complete98/49418 videos before selecting ONE by developer sequence
IoU. Same checkpoint then full LasHeR245/220703 +RGBT234234/116649, allfive native
metrics/all31attributes/fourreferences/5000pairedCIs and state mechanisms;
retire11ownunused heads after all consumers, keep selectedbest and dependencies.
No >=3positive-seed gate, no TEST tuning, no dataset move or power/temp actions.
Actual update 2026-10-06T11:35:25.855066+08:00: deployed source/CPU witness and allfour real GPU sanity/full60/840 fits PASS; original3801966 in FULL98_BEST. Eightfull98/oneweight/bothnative pending, not formal success.


## 2026-10-06T14:17:32.093926+08:00 Current-future complete actual native closure

Four60/840, eightfull98, one epoch0 both245/220703+234/116649 complete. Five78.016094/74.383784/62.072660 +92.866361/70.022381 exactgrossparent, no trainedCgain, all+2false. All31attrs/81rows, four5000pairedrefs/mechanisms/efficiency. Original3801966 andobserver14340/91871 CLOSED. Elevenownunusedheads2539938B retired by original queue; selected/dependencies retained.1998originalartifacts copied/noNNorreportreplay. Newrelation source/syntheticCPU only, realGPU/controller pending. Rawlabel interpretation corrected: IoU==0nonTarget, partialoverlapunknown; no halfGTsupport among zeroIoU. GoalACTIVE_UNMET; onebest enough.


## 2026-10-06T14:31:26.589644+08:00 Reviewed candidate-relation raw ABC pipeline

Four matched ordinary/relation ABC configs lr1e-5/1e-4, batch336/seed42, same nine TRAIN caches2576states/881seq and196queries/98developerVAL. Pretrained fullGOLA/C1/old4/frozenmotion retained. All A/B/C and newrelations train. Oracle region labels,budgeted actions,action writes,gross selection, including objective search ranking aligned to gross. Real2ep sanity with gradients/updates/reloads; actual saved initial.pth full98 parent byte/decision parity on four frame-balanced shards; then four60ep/480updates and eightfull98 best/last candidates. Same best checkpoint both formal datasets only after full-video selection. If no parent improvement keep incumbent and explicitly reference completed parent formal results without identical NN replay.19/20 unused OWN weights cleaned only after consumers. Review scope COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE SOURCE PASS, realGPU/runtime/effect not yet accepted. These existing states/three-step frozen continuation labels are not new independent events or new trusted-geometry/long-risk labels. No seed-stability requirement.


## 2026-10-06T14:39:08.163310+08:00 Reviewed candidate-relation raw ABC pipeline

Four matched ordinary/relation ABC configs lr1e-5/1e-4, batch336/seed42, same nine TRAIN caches2576states/881seq and196queries/98developerVAL. Pretrained fullGOLA/C1/old4/frozenmotion retained. All A/B/C and newrelations train. Oracle region labels,budgeted actions,action writes,gross selection, including objective search ranking aligned to gross. Real2ep sanity with gradients/updates/reloads; actual saved initial.pth full98 parent byte/decision parity on four frame-balanced shards; then four60ep/480updates and eightfull98 best/last candidates. Same best checkpoint both formal datasets only after full-video selection. If no parent improvement keep incumbent and explicitly reference completed parent formal results without identical NN replay.19/20 unused OWN weights cleaned only after consumers. Review scope COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE SOURCE PASS, realGPU/runtime/effect not yet accepted. These existing states/three-step frozen continuation labels are not new independent events or new trusted-geometry/long-risk labels. No seed-stability requirement.


## 2026-10-06 deployed-current relation policy data aggregation

The selected epoch5 model full60/480 original training, all eight full98 and both native benchmarks are complete; all five +2 targets remain unmet. The fixed-model full98 diagnostics completed at21:44:45: selected .744415038, no-extra .743650054, pause-off .742451954. Both controls lower point estimates; pause-off only2 developer sequences differ, and does not disable identity-memory verification. Keep these negative controls and protected parents.

A concrete collector mismatch is now identified: its frozen own prefixes/future tracker inherited weighted search, while the evaluated selected policy explicitly uses gross. In addition the collector accepts only ABC_recoverability and cannot load the actual selected ABC_candidate_relations. Extend these exact two paths; leave previous collection/default behavior and model architecture unchanged. This is a state-generation consistency fix, not established cause of the performance gap.

Next collection milestone uses the fixed restored epoch5 current policy, explicit gross/action/own continuation and original disjoint881/98 split. From2576 prior TRAIN queries sample256 distinct states with seed42 (128 failed,64 ambiguous,64 normal according to prior TRAIN keep IoU). Preserve all196 existing reused developer queries. Execute four TRAIN and four validation shards, preceded by one-query-per-card TRAIN and validation sanities. Validation sanity must compare all causal history boxes to the archived unrounded full98 selected-policy decision boxes (float32 exact); this is deployment-prefix parity, not GT accuracy. Current/future labels continue to use real TRAIN/developer GT; future inputs are copied only after the current observations are locked. Same complete GOLA/C1/motion, all selected A/B/C and relation parameters retained/frozen for data generation; at most one extra crop per deployed step, six teacher query alternatives disclosed. No TEST, new weights or retirement. Runtime memory remains four slots/modality and eight geometry observations; offline prefix arrays bounded1024. Estimated4 GPU-hours total pending real sanity timing; disk gate20GiB; no power/temperature operations. Long polling240seconds/near ETA.

Next training after collection must explicitly aggregate distinct policy states rather than silently collapse the new rows by a generic prefix-policy string; this loader change/full module fit is not yet launched or claimed complete. Use one best checkpoint selected before both native tests, and retain protected parent/current checkpoints until consumers close. No three-positive-seed requirement or all-module efficacy claim.


## 2026-10-06 current-policy aggregate full-training queue

Predecessor is actual current-policy collection535051: both8sanity queries/developer unrounded-prefix parity passed, four TRAIN workers536876-79 last actualLIVE22:13:43. Do not duplicate or restart collection; sole observer36780 next22:32:27/240seconds/near measured ETA. The downstream queue must wait for this owner to exit and its exact COMPLETE receipt, then verify free cards/disk before any fit. This extends the existing explicit four-worker phase controller, preserving user cadence and stopping on primary failure (no unchanged OOM retries or a new screen framework).

Minimal training fixes: current ABC_candidate_relations checkpoint must strict-load all parameters including relation block; the added aggregate-policy-states flag distinguishes each sequence/query by its actual prefix checkpoint rather than the generic prefix description. Old default loading and replay behavior stay unchanged. Current files use the same own future-policy schema/horizon3, real futureGT onlylabels. Current256 TRAIN queries overlap old2576 times by design but come from another deployed policy: aggregation has2832 policy states and2576 unique sequence/query times. It is not2832 unique videos or queries. Old source teacher defaultweighted versus new currentgross is disclosed and preserved; this comparison tests new state generation, not one isolated architectural factor.

Four arms all initialize the SAME protected epoch5 relations, train A/B/C+relations and freeze complete GOLA/C1/frozen-motion exactly as before; seed42/B336/oracle/budgeted/action/gross. Old9 roots:2576 states/72epochs/576updates at lr1e-5 and3e-5. Aggregate old9+current4 roots:2832 states/64epochs/576updates at the same2learningrates. Matched update count8x72=9x64=576; each epoch completes the whole data, botharms>=60epochs. Four2epoch full-data capacity/gradient/reload sanities precede full runs; no architecture replacement or zero relation reset when continuing existing model. All arms validate on196 newly collected current-policy developer states, CPU merged/reordered to original196 queries and strict NPZ-reloaded. All881TRAIN/98developer sequences remain disjoint.

Retain best and actual endpoint through eight FULL98/49418 runs in two four-card waves. ChooseONE checkpoint by developer sequenceIoU before bothnative245/220703 and234/116649. ALWAYS obtain allfive/31attribute settings/curves/paired intervals via existing fullnative engine, regardless of parent gain; no TEST selection/seed-stability gate. Only after native consumers close retire own unselected best/last andsanity weights, keep selected plus old4/currentparent/GOLA/C1/DINO/motion dependencies and every raw negative result. Disk20GiB budget, measured existing output capacity>2.6TB; actual free rechecked after predecessor ends, no dataset moves or power/temp operations. Estimatedfullfits80-100min plus eightfull98~65min and native~75min (shared IO, no guarantee); observe at actualsanity-derived near-stage ETA then180-300sec.
