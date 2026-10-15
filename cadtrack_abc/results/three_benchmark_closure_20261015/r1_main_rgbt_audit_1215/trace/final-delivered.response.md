**WARN，限定数值报告范围内 0 阻断。** 1437 项确定性检查通过，2234 份输入前后哈希不变；same-family/provisional，实际模型／路由／推理强度均 UNATTESTED。

| 数据／模型 | Precision % | Success % |
|---|---:|---:|
| 主模型 RGBT210 | PR 87.4103549305 | SR 63.6677661266 |
| 主模型 RGBT234 | MPR 89.1595470178 | MSR 65.9338861290 |
| pure CAD RGBT234 | MPR 89.8813070751 | MSR 66.3103044038 |

RGBT234 主模型减 pure：MPR −0.721760 点，95% 配对序列区间 [−2.269729, 0.817624]；MSR −0.376418 点，[−1.448652, 0.705736]。5000 次、seed 20261015；不支持提升结论。

实际 NumPy2.2.6 完全复现所有分数、属性、曲线、CSV 和延迟。首次 NumPy1.26.4 的 SR/MSR 边界差异及退出1原始记录完整保留。选中权重 SHA 匹配，但不能证明实际加载、完整运行时源码或神经推理。主报告“两 GPU”计时文案错误，实际两个分片同在 GPU2；不能据此声称加速。整体研究目标和三个模块的独立有效性均未认证。

[审计报告](C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_rgbt_formal_audit_20261015/EXPERIMENT_AUDIT.md) · [机器结论](C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_rgbt_formal_audit_20261015/EXPERIMENT_AUDIT.json)

同目录已保存复算 CSV、bootstrap、输入哈希、执行回执和完整审查请求／响应 trace。
