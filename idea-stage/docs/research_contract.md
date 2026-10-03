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
