# C1 部署前代码审查（2026-10-02）

Reviewer: Codex gpt-6-astra，reasoning=max，fresh context，任务 c1_code_review。
review_independence: same-family；acceptance_status: provisional。

结论：PASS，可进入 C1 sanity；无 BLOCKING。

核对：DINOv2→官方 GOLA LoRA 合并路径、完整在线模板与 head、base eval/冻结、head-only optimizer；Hann 原赢家保留和局部峰/NMS/距离去重；归一化 XYXY GT IoU 与 BCE/排序方向；有效槽位与 pair mask；序列 train/validation 分离、固定验证采样、历史早于当前、GT/未来未输入新头；JSON/CSV/checkpoint/receipt 与 C1 边界说明。两 Python 文件 AST 通过，JSONL 真实换行。

NONBLOCKING：默认 validation-steps 为64而正式计划32。执行者已改默认32，实际正式launcher也显式传32，sanity显式传2。不存在阻塞修补，不需要额外兼容/防御逻辑。

审查没有运行服务器 GPU。真实 checkpoint load、有限 loss、参数更新和显存必须由 sanity 验证，不能把代码 PASS 当作实验 PASS。完整 A/B/C2/C3 未实现，不在本次 C1 PASS 的贡献范围。

## 22:33 RGBT234 actual GT scoring correction — PASS
Reviewer: fresh gpt-6-astra / max, task rgbt_gt_source_review; same-family / provisional.

BLOCKING: none after correcting the initial direct-gt_path attempt. That attempt failed the CPU check because the class additionally reads ../attr_txt; it was never deployed. Final code initializes default official attribute groups, then uses original initial_gt_file to load actual visible.txt / infrared.txt and records the actual GT root. No fallback or metric formula changes.

NON-BLOCKING: no new code issue. Author calibration and previous packaged-GT scores must remain explicitly separate protocols.

Evidence: native loader verifies all234seq, each modality116649frames, orangeman1 both920; native radar reads precomputed attribute groups. All print_metrics AST are identical to original, syntax/diff checks passed. monitor_core identifies RGBT234 completion without actualGT log as needs_actual_gt_scoring. Inference does not depend on these files; wrapper calls scoring only after inference.

Reviewer did not deploy, edit, or stop jobs. Executor deployed evaluation.py and monitor_core after PASS, py_compile succeeded, then started CPU-only rescoring of existing full234seq predictions. Packaged-GT logs/markers retained with packaged_gt_obsolete suffix. Actual result receipt still pending at this update. No inference rerun.
22:35真实部署验证完成：两份全量预测234seq/116649frames通过每序列GT长度/四列/有限值检查，评分成功，log actualGT path确认。baseline MPR91.778858/MSR69.232593，C1 MPR91.817009/MSR69.233890；差+.038151/+.001297百分点，整体基本持平。原包GT日志保留，无推理重跑。
