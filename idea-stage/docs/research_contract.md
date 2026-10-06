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
