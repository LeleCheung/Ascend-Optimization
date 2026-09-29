# 最终 best 独立复验与 Agent 复测请求

## 决策与范围

每个候选 SHA 和冻结评测契约在最终输出前必须完成一次独立复验；Agent 可根据已记录的双边计时申请额外复测。使用现有 KGS evaluate，它在受设备队列调度的新隔离子进程中执行，不复用搜索轮的时延，不改变 wire Protocol。复测仍运行完整的原始正确性和计时计划；不把 adapter 工作区切到另一个 native 测试口径，也不改变 Gems 的 core 计划。

这是 KG 的独立开发分支 `feat/final-best-retest`，基于已合入 pytest review/candidate admission 的 dev；无需修改 KGS 源码，但 Server 必须提供 candidate_admission capability。不是发布或兼容清单变更，旧部署缺 capability 时不能绕过。

## 执行与证据

1. 搜索评测快照额外冻结 bundle 身份、评测 settings、Server 已暴露的运行环境身份和完整结果 hash。复测校验 candidate、definition/workloads、benchmark fingerprint、配置和 Server 身份；原快照不完整或发生漂移时返回 NEEDS_RETEST，不能假定旧环境相同。
2. `request_retest(round_num, reason, evidence_workload_uuids)` 只接受既有通过轮次及其 workload UUID。UUID 是问题证据锚点，不是测试子集；reason 是 Agent 的诊断说明，不是已验证的根因。Agent 无法通过此工具传入时延、判定、容差或计时参数。
3. 优化停止后，KernelOptimization/SimpleOpt 外层自动调用最终复验，再决定是否进行 profile、distillation 和成功输出。Agent 提前返回成功不能跳过复验。追加复测不消耗搜索轮，也不调用 finalize_round，不更改历史 best 选择或原始时延。
4. 每次请求先持久化记录再访问 Server，完成后保存完整结果、逐 workload reference/candidate 时延、两侧变化比例和确定性结论。记录位于 `.kernelgen/retests/<candidate SHA>/<contract SHA>/attempt-XXXX.json`，最终摘要为 `.kernelgen/final-verification.json`。按候选 SHA 和契约合并预算，重新提交相同代码为新轮次不能清除失败证据。

当前成本上限是一次自动最终复验，加最多两次 Agent 额外请求。并发复测使用 workspace 内锁；中断的请求保留，自动恢复不会静默重复未知状态的设备请求。已有最终请求时使用最新复测结论，不挑历史最高加速比；换候选 SHA 后重新验证。复测不会改写搜索 receipt，不需要对已经准入且源码/策略未变化的候选重复源码扫描。

对已经导出的 best 追加复测时，先将原输出置为 NEEDS_RETEST 并清空公开加速比；随后恢复 workflow 才能重新导出最新结论。多 Coder 的 KernelGen epoch 汇总和断点恢复只选择与当前 ledger 赢家 SHA/round 一致的最终确认结果，并按复验分数排序，不能从搜索 ledger 恢复旧高分或晋升未确认候选。

## 判定

| 结果 | 含义与后续 |
|---|---|
| PASSED | 正确性有效通过，测试身份及 skip 结果一致，双边时延完整且未出现超过阈值的变化；最终加速比从最新完整复测的逐点时延重新计算 |
| FAILED | 明确的数值/候选准入失败；保留证据交候选维护方，不能用同 SHA 的反复抽样洗掉失败 |
| NEEDS_RETEST | 时延显著漂移、证据缺失、环境或测试计划变化、运行错误或中断；不是候选低性能，也不是新增 BLOCK 原因 |

初版对每个有效计时点分别比较 reference 和 candidate：新/旧时延比超出 `[0.5, 2.0]` 为 TIMING_DRIFT。这是保守的异常提示，不是统计置信区间，不声称覆盖轻微退化。第一次独立复测与原搜索轮比较；若申请后续复测，则使用最近两个连续独立测量比较，仍核对原始测试集合。两次独立测量一致后可以采用最新实际时延，即使它证实原搜索加速比被高估；不使用旧 headline 或挑选最大值。所有尝试保留，最近一次失败不能回退到此前通过记录。

当前 PASSED 沿用正确性和有效计时口径；最终 `best_geo_mean` 可能低于 0.8，优化合格仍由现有汇总阈值另判。`search_best_geo_mean` 保留原搜索值；`final_verification` 指向复验结论。未确认时输出 best_geo_mean=null，不蒸馏成功经验；候选源码仍保留供诊断。搜索 ledger 和 `.best_kernel.py` 继续表示搜索过程产物，不能脱离最终输出把它们视为复验已通过。

TIMEOUT 或 SUSPECTED_DEVICE_ERROR 保留实际结果并检查 scheduler；不自动换卡或重复复测。搜索以这类状态终止时，也不会为了验证旧 best 再自动发起设备工作。由操作方处理设备/超时预算，不交 Coder 修补环境。

## 能力边界

该机制不证明 reference 数学语义，也不替代 pytest review 的候选注入/baseline 隔离检查；相同的系统性错误可能在多次复测中重复。冻结的是测试计划和 Server 已暴露的环境字段，不证明未记录随机输入内容或镜像内部文件完全相同。KGS 当前结果提供每次 evaluate 的逐 workload 汇总时延；保留每次独立 evaluate 的原值，但不能声称已经得到计时器每一次内循环的原始采样。

旧快照没有 settings/environment/result hash 时不伪造复验身份；新工作区才能获得完整新快照。源码明确违规由 preflight 负责；正确性与设备异常保留所属阶段，复验不扩大 Coder 的职责。

## 验证入口

Host 用例见 [test_retest.py](../../../tests/test_retest.py)，覆盖旧 reference 异常、双边记录、缺失/重复/改变的 workload、数值失败、同代码重复请求、预算、取消、断点恢复、环境变化和 workflow 成功输出门禁；设备验证另外记录实际版本与运行产物。
