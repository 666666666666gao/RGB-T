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

## 23:55 C2 / all-core collector / branch batching — PASS
Fresh gpt-6-astra/max reviewers c2_bounded_state_review and full_metrics_and_batch_review, same-family/provisional; no BLOCKING after age bound fixed to <W including birth. Checked branch-private committed/pending/search state, anchorimmutability, pre-updateparent forks, bounded lifetime and count, same-frameGT external switch diagnostics, disjointoffset0/8 shards, batch3 tensor/mask/crop correspondence, original metric/GT/attribute source and LasHeR NPR→PR→SR ordering. Real serial2seq sanity and16seq heldout diagnostics completed; latter NEGATIVE (.710145→.696210) withC2=box-only, must not claim repair efficacy. Real batched3 vs3single all numericalmaxdiff0, P≤3/pending≤4, anchorunchanged. GPU inference coverage remains full245/234 gates, not smoke.

## 23:57 C3 rollout utility — PASS
Reviewer c3_rollout_utility_review, fresh gpt-6-astra/max, same-family/provisional. BLOCKING none. Source directly read: branch_utility.py, collect_rollouts.py, train_branch_utility.py, bounded_recovery.py, candidate_learning.py, evaluate_online.py. Confirmed881/98 unique disjointtrain/val; queryq+H bounds; pastselfprediction; current768/10/9feature snapshot beforefuturedecode; GT onlyIoU/wrongwrite labels; copieddeques/privateupdates; alternatives startcommittedtemplate; fixedC1 continuation; futureHmean-.1wrongwrite/current+H; correctvalidpairloss sign; checkpointminimum internalvalregret; finite checks. WARN zeroresidual ranksqualitywithoutHann, thusretainepoch0vsbaseline comparison; explicitvalclips64/sanityclips2 required.
Real deployment sanity23:54 both2clips completed train8validactions875.967MiB /val9actions924.504MiB. Training2epochs/2optimizersteps newparamschangedtrue/base0, finite loss/grad; valMSE.345938→.340665, utilityunchanged. Noofficialtrackingclaim. Initialteachertrain128/val64,H3/history≤256,batchclips8 launched23:57:21GPU2/3.

## 23:59 full candidate/update timelines — PASS after concrete precision fix
Reviewer full_candidate_metrics_review, fresh gpt-6-astra/max, same-family/provisional. Readcandidate_learning.py/evaluate_online.py/collect_candidate_metrics.py pluscorecollector andconsumers. Originalforward remains tensor-onlydict; newextract_with_output returnscandidates+denseoutput fromsameforward, baselinepostprocessorunchanged. OnlyfirstGT initializes; timelineCPUserializationafterframetimer; allseq/no-smokereceipts/exactoriginaltrajectorycomparison; XYXY/XYWHvalidslots+actualmodalGTmax; recall/rescue/harm denominators; prior-template-sourcepollution withunknownlabels excluded; qualitybinaggregation correct. BLOCKINGnoneaftercopyingactualtrackedbox intotimelinechosen slot, becausebaselinefloat32scaling differs fromgenericdoublecandidate mapping. Numericalprecisioncase evidencedinsource, nofallbackortrackingchange. SixfilesAST/diffwhitespacePASS. GPUcorrespondencesmokerequirednext beforefullaudits; baselineinstrumentedtimeincludesextraretrieval, oldfullspeedusedforcomparison.

## 01:45 zero-residual trajectory control — PASS
Fresh gpt-6-astra/max reviewer zero_head_equivalence_review; same-family/provisional. No BLOCKING. Read source, actual server data, CPU arithmetic; reviewer did not execute tracking. Each comparison initializes independent tracker/templates/crop state; frozen eval fullpretrained and zero last layer, seed42, no optimizer. Later annotation fields exposed by wrapper never enter tracking/decisions; corrected doc wording and removed unused import. Completion/NPZ/rawXYWH differences/update/selection counts correct; official accuracy false.
Server inspection: train cache and881/98 split confirmed; both complete ordered raw/cache modality paths exactly equal for six fixed videos, no omitted/extra frames. Offset0:10rightblackboy333/initXYXY855,424,925,559;1boygo224/734,203,860,479;2girlgoleft138/545,132,572,186. Offset3:2outdark122/277,158,304,213;2rdtribike135/568,1,604,44;3whitemen449/677,138,695,178. Groups695/706frames=1401, disjoint, every init.txt first row xywh→xyxy exactcachedinitializer. No cache rewrite/extra filter required.
NONBLOCKING interpretation: float32*224 beforedouble vsdouble*224 CPU probe5.72e-6px; logit/sigmoid reconstruction up to5.96e-8 probability. Diagnostic measures combined real implemented-path difference, not exclusively scaleprecision. np.round(...,3) compares numeric rounding, not serialized-text bytes. GPU mapping CUDA_VISIBLE_DEVICES=2/3 required since scriptcuda:0. Runtime evidence still pending; no precision correction deployed.

## 01:53 precision correction and reporting revision — PASS
Same fresh gpt-6-astra/max reviewer, same-family/provisional revision. Real v1 controls1401frames, zeroaltchoices andequalupdatecounts eachvideo but45roundedframesdifferent, 1boygo max2.004583511px. Count equality does not imply equal update timing. Minimal C1 float32 multiplybeforedouble matches actual originalpostprocessor exactly in CPUcheck; oldprobe max1.1444e-5px. Weights/Hann/confidence/threshold/state unchanged. V2rawsix-video equivalence stillmust be run before correctedfullbenchmark.
ReportingBLOCKING identifiedandresolved beforedeployment: self-reference cannotassert independent originalrun equivalence. Collector now saves referencepath,separate-runbool, exactreference matchafterasserts, and exactoriginal=False forself (independent equivalence notestablished, not measuredmismatch). Generic reference/source wording andCLIhelp updated. All fullcoverage/actualGT/timeline-selected checks unchanged. No extra repetition needed. Unselectedtimeline boxes retainpriorconversion andchosenactualboxoverwrite remainsvalid.
