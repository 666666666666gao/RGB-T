# R3 顶层失败状态：限定修复确认

**PASS_SOURCE_ONLY_N1_FIXED；BLOCKING 0，剩余已知 NONBLOCKING 0。**

仅比较封存 delta 的 `dense_main_pipeline.py` 与原35项封存中的对应文件。确认第83–84行恰好增加两行：子进程退出码存在非零时，把顶层状态写为 `DENSE_R3_MAIN_PIPELINE_FAILED`。位置在现有 `phase.update(...)` 后、`save()` 和原有停止断言前；因此失败状态会随阶段退出码一起保存。全部子进程成功时继续原有路径。

实际 stdlib 检查确认：文本差异只有这两行；删除该 AST 节点后，两份 AST 完全一致。没有 retry、try/except、fallback，也没有训练、选择、标签、命令、等待或停止语义改动。N1 已在这份 delta 中解决。

delta SHA256：`518f4b1ac2828f482b97da755d6e7f25d43b74f61272141b6e979f7ba8e4a42a`。

实际检查使用 `E:/python.exe`，退出码 **0**；[diff](failure_status_delta.diff)、[stdout](failure_status_delta.stdout.txt)、[stderr](failure_status_delta.stderr.txt)（空）及[命令/退出码 receipt](failure_status_delta.receipt.json)均保留。没有辅助检查失败。

原35项全部仍匹配原manifest，原manifest、原 `SOURCE_REVIEW.md/.json` 及其时间戳版本前后均未变化。delta与DELTA_SCOPE自身也未变化；见 [before](fix_preservation_before.json) 和 [after](fix_preservation_after.json)。原 `PASS_SOURCE_ONLY_WITH_NONBLOCKING_NOTE` 报告保留，本文件是限定修复追加结论。

请求 Astra/max；没有宿主实际路由证明，`actual_route=UNATTESTED`，`same-family/provisional`。未 import Torch 或项目，未执行项目函数，未读取服务器、NN、GPU或真实图像，未启动任何NN。原报告的依赖覆盖边界及真实运行义务继续适用；本结论不等于 runtime 通过，不产生正式指标或 goal 完成声明。
