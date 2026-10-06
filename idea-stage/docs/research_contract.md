# 研究约定

固定预算下保留纠错机会是待验证主张，不是已有实验结论。新方法必须以完整 pretrained GOLA-B 为基线，不能删去在线模板再把恢复该能力算作增益。

- C1 仅验证候选质量/排序头能被训练，以及基线失败时被保留候选的诊断上界。crop-based held-out IoU 不代替在线 PR/SR，不宣称状态修复。
- A 的主张需等容量压缩比较；B 的主张需等候选与搜索预算比较；C 的主张需完整在线分支/状态修复，并报告错误切换。
- 数据：LasHeR 官方 train 训练，train 内序列划分开发验证；test/RGBT234 仅正式评测。只使用实际数据集真值作标签和评测真值。
- 测试只能截至当前帧，不能修改过去输出；训练未来监督必须与推理输入隔离。
- 所有分支、额外主干前向与扩大搜索均计入完整推理成本。来源、路径、配置、随机种子、负结果和失败记录必须保留。
- 尚未验证对 P-SRM、KeepTrack、RecheckTrack 等近邻的实质增益，不能宣称新颖性或 SOTA 已证实。

## Integrated ABC implementation acceptance (2026-10-03)
Goal remains ACTIVE until A/B/C are causally integrated and the full same-protocol method exceeds frozen complete GOLA-B by >=2 percentage points on ALL LasHeR PR/NPR/SR and RGBT234 MPR/MSR. Current observed baseline is76.598449/73.101973/61.037425 and91.778858/69.232593; required main scores78.598449/75.101973/63.037425 and93.778858/71.232593. Complete all245/234 sequences, all31 attributes including declines, harms/recovery/candidates/branch-aware contamination/forecast calibration/latency/memory, fixed-checkpoint uncertainty. No test threshold/checkpoint selection, no altered GT/protocol, no substituted subset metric.
New source A=protected separate pre-attention identity anchor+4slots/modality learned gated discriminative compression of full causal sampled prefix, B=3mode/H3 diagonal Gaussian joint trajectory conditioned on A+recent8 actualpredboxes/timestamps, C=current quality/future value/wrongwrite cost with rejected-candidate truth supervision and coupled score. New onlineABC3branches/W5 keeps private committed/pending learnedmemory/template/motion, previous outputs immutable; memory/trajectory state reset test must demonstrate real separation. These mechanisms have been IMPLEMENTED in source, not yet source-approved/GPU-trained/fullbenchmarked at this contract update. ExistingC1formalresults and negativeC2/C3 retained.


## 2026-10-06T07:07:09.435730+08:00 selective-state-commit execution update

Actual four-GPU query M0 PASS, full1789labels collecting original3338360. Original3351735 queued: four2epoch fit sanities, fourfull60epoch C-extension fits (H3/H32 × lost0/.1, B128 LR.001 seed42), all98video selection, onehead forboth245/234 native/full5/31attrs/4pairedreferences. A/B+completeGOLA/C1/old4 remain active/frozen in this prototype; not jointABC retraining. Actual consecutive known-state events get totaltrainingweight1. Two27/32frame videos explicitlyH3-only; H32 doesnot inventfuture labels. OptimizerVAL exclusion, nofutureGTinput, no TEST checkpointchoice; onebestmodel sufficient, allfiveGOLA+2 stillUNMET. Humanhandoff only docs/HANDOFF_20261002.md; authoritative machine snapshot refine-logs/runs/selective_state_commit/current_execution_snapshot.json.


## 2026-10-06T07:45:41.497027+08:00 pre-fit endpoint-selection correction

Actual waiting-only3351735 retired with no fit/children/updates. Replacement3423188 waits unchanged3338360 collection; no NN collection/training replay. Four full60 fits preserve cached-best and actual60endpoint, strictreloadboth. All eight candidates each full98/49418 in two4GPU waves; selectONE by developer sequenceIoU before samehead bothnative/all5/31attrs/4refs5000CIs. Retire11 own unused heads onlyafter all consumers. Recipe/data/model unchanged, no>=3positive-seed gate, targetall5GOLA+2 remainsUNMET.


## 2026-10-06T10:20:55.846949+08:00 actual full-training/internal closure, native ongoing

Four C-extension fits each60/840 (not jointbackbone training), eightfull98/49418 complete; all learned checkpoints below frozenold4/gross. One epoch0 locked before bothnative; actual98prediction text equalparent, no training gain. TRAIN1646event1691queries/881seq and615write-pairqueries; threeeventroles overlap. Actual16CPUcachedforwards show TRAINpositive/VALnegative selectedutility despitepositive predictedutility. History244CSV/plots and report metadata copied; no NN/report replay. Currentoriginal3423188/native3643976-79 lastactual09:59:49, sole22853 next10:53:18 then240. Five formal new metrics pending; all5+2UNMET. One best sufficient, old4/dependencies protected.


## 2026-10-06T11:09:59.304261+08:00 actual full native closure; goal still unmet

Four60/840 fits, eightfull98 and SAME epoch0 both245/220703+234/116649 actual-GT complete. Five78.016094/74.383784/62.072660 and92.866361/70.022381, allbaseline+2false; exactgrossparent scores/no learned C benefit. Las threeCIpositive/RGBtwoinclude0; all31attrs/81metricrows/curves/four5000pairedrefs/mechanisms/efficiency copied3838files.11ownunusedweights5303694Bretired by original queue afterconsumers; bestepoch0 anddependenciesheld. Original3423188 andobserver22853 CLOSED; no replays or activeNN. Negative training and repeateddeveloper/test limitations retained. GoalACTIVE_UNMET, one best sufficient.


## 2026-10-06T11:35:25.855066+08:00 current/future reward alignment actual execution

SOURCE reviewPASS/provisional + actual deployed CPU default-exact witness + four GPU2epoch sanityPASS + four full60/840 fits strictbest/last reloadPASS. Original3801966 nowFULL98_BEST; eightfull98 selection and samehead bothnative/5+31attrs/4refs5000 pending. Same completed1789/event-weighted corpus; A/B/GOLA/C1/old4 frozen, C-only MLP119725 vslinear4588 H3/H32. Allfive+2UNMET; one best sufficient. No NN replay/oldqueue restart/dataset move/power-temperature operation. See current_execution_snapshot in state_commit_current_future and ONEHANDOFF.


## 2026-10-06T11:48:34.655819+08:00 sealed four-fit CPU diagnosis

Actual16cached comparisons/32CPU-head forwards/strictreload PASS; allfour last60 TRAIN true utility positive but reused developer98 negative and predictedpositive. Linear4588 also negative; lossdecrease notdecisiongain. All cachedbest0. Frozenparent futurelabels, unknownmasked; not fullvideo/native efficacy. Collected244historyrows/config/completion/plots, no weight/NPZ copies or liveNN/GPUqueries. Original3801966/sole14340 first12:07 unchanged, all5+2ACTIVE_UNMET.


## 2026-10-06T12:30:49.481953+08:00 actual first4 complete full98 acceptance

All4cachedbest0 complete98/49418 and98TXT exactparent each, Cintervention/geometryhold0: no newtrainedgain. Actual12:11 original3801966 and newNN3867094-97 live FULL98_LAST. Same60 endpoints all98 pending, then oneweight bothnative/all5/31attrs/4refs5000. Sole14340/session91871 next2026-10-06T12:45:56.393785+08:00, then240; no replays/liveNNsource edits/cleanup. smoke_onlyTrue isvalidation-split flag, actualsequence/frame limits0; fullinternal notnativeTEST. All5+2ACTIVE_UNMET; Ccache917 encodedfeatures isnotABCraw-input training.
