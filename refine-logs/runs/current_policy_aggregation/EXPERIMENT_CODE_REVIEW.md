# 当前策略聚合训练源码审查

- 状态：**PASS**；没有发现具体阻塞缺陷，无需修改源码。
- Scope：`COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE`
- Run scope：`DEPLOYED_CURRENT_POLICY_AGGREGATION`
- 时间：2026-10-06T22:41:02.910308+08:00
- 审查者：`/root/current_policy_aggregation_review`，按 experiment-bridge CODE_REVIEW 的独立上下文审查。
- 请求模型/推理：`gpt-6-astra` / `max`。后端身份未独立验证；`same-family`、`provisional`、`runtime_verified: false`。

已直接阅读 trainer diff、新队列、最后一节计划、已有 native 引擎及私有部署脚本，并用本地真实配置、采集计划、8份已完成 sanity 回执和历史 NPZ 做 CPU 复核。没有运行 SSH、前序轮询、模型前向、优化、部署、TEST 或权重清理。

## 已核实

1. 旧9个 TRAIN 源有2576个唯一序列/查询时刻，覆盖881条序列。新增256个查询时刻全部属于旧TRAIN，分层为128失败/64模糊/64正常。按实际 checkpoint 策略标识聚合后是2832个状态、2576个唯一时刻、2种策略。旧/新共用缓存契约相等；881TRAIN与98VAL序列交集为0。默认加载行为保留。
2. `ABC_candidate_relations` 在模型构造时建立 relation 模块，随后 strict-load 整个 checkpoint 并检查全部张量相等。继续训练路径没有新增零初始化 relation；分组、梯度、变化和保存回执均依据实际 `has_relations`。A/B/C及relation可训练，C1固定，GOLA和motion沿用冻结缓存/评测路径。
3. 四个49条VAL分片按原196个查询恢复顺序。实际计划上的索引为完整双射，重排结果与原jobs逐项一致；运行时还会检查所有数组有限、重新加载NPZ逐元素相等。当前完整新NPZ尚未在本地读取。
4. 四臂均为seed42/B336、相同epoch5 parent、oracle/budgeted/action/gross；LR为1e-5与3e-5。预算如下：

| 数据 | 状态数 | 完整epoch | 每epoch更新 | 总更新 | 2epoch sanity更新 |
|---|---:|---:|---:|---:|---:|
| old | 2576 | 72 | 8 | 576 | 16 |
| aggregate | 2832 | 64 | 9 | 576 | 18 |

四个完整数据2epoch sanity必须全部完成梯度/参数变化/C1固定/最佳及终点strict reload检查，才开始完整训练。完整训练每个epoch覆盖所有状态；这匹配更新次数，样本累计呈现次数分别为185472与181248。

5. 两个四卡波次评测全部8个best/last checkpoint的98序列/49418帧。GT计算的sequence mean IoU选择一个checkpoint并先写入锁定文件；之后无条件使用同一权重完成两个native数据集，不依据parent gain省略测试。已有引擎契约相符，覆盖LasHeR245/220703与RGBT234234/116649、五指标、19+12属性、曲线及相对GOLA/C1/old4/gross_parent的5000/seed42配对序列区间。
6. 原535051退出且收到精确COMPLETE回执后才检查四卡空闲、>20GiB空间和新缓存。等待240秒；不新增采集器或观测器。只有native同步返回并核实相同模型后，才枚举本轮16个sanity/full的best/last权重，保留所选1个，删除其余15个；保护父模型和全部依赖、负结果、日志。
7. 私有部署脚本只暂存/发布训练文件、计划和本次目录；实际本地main分支、暂存区为空。现有采集器/model源码及MANIFEST不在发布清单。远端编译/已实现的源文件一致性检查通过后只启动一个依赖535051的owner，`child.poll()`检查存活；`requested_first_phase`没有伪装为读取到的进度。
8. trainer、队列、native、私有部署器及其嵌入远端源码AST通过；CRLF兼容的`git diff --check`通过。本地离线NumPy检查了真实配置计数、策略身份、分层、8份sanity配置/回执、VAL重排及更新预算。真实历史own-policy NPZ的horizon3及输入/标签dtype、shape、有限性一致。

## 结论边界

这是源码PASS。新训练的运行、峰值资源、当前完整新缓存和native性能仍需队列实际完成后验证。审查仅消费已有22:13:43观测；没有宣称前序已经结束。恢复epoch5的历史回执证明其指定范围内的developer预测和决策一致，不等于原权重字节身份。

旧缓存的隐式weighted与新checkpoint显式gross/action/own都保留并已披露；本次比较是更新状态生成/聚合与学习率控制，不是隔离的架构因果结论。196/98开发集重复使用；配对区间描述固定权重的序列变化，不代表训练seed稳定性。不增加三正seed门槛，也不宣称新性能或五项+2目标达成。

完整证据、源码位置和运行限制见同名JSON。审查没有修改生产源码。
