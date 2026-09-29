# 统一测试契约审核

OperatorOptimize 的新计划为 `prepare_catalog → review_tests → optimize → code_review`。Native/Gems、本地/已安装四类输入使用相同审核策略：默认调用只读 Reviewer，只有显式 `skip_review=true` 才不调用模型。安装式来源、Gems 来源校验、抽取阶段通过的审核记录均不是免审依据；来源一致性校验保留在 prepare，不把它当作语义审核。

CLI 使用 `kg run ... --skip-review`；YAML Batch 的 defaults 或 operator 使用 `skip_review: true`；Python 在 OperatorOptimizeInput 顶层设置同名字段。默认 false，`--no-skip-review` 可以明确关闭跳过。这个选项不跳过优化后的代码审核，也不跳过目标执行验证。

## 审核证据与结果

prepare 从目标 KGS 获取冻结契约和审核源码。`/status.capabilities.operator_contract.test_sources=true` 声明按需导出能力，`POST /operator-contract` 的 `include_test_sources=true` 返回源码文本；KG 保存原始 `test-sources.json` 作为冻结事实。每次审核从该快照派生逐文件的可读文本和带来源路径、SHA-256 的索引，长文件可分页读取；审核回执绑定实际阅读文件的摘要。代码审核复用这些可读证据，不让模型重新读取包含超长单行字符串的原始 JSON。Native 证据是 oracle、workloads 和伴随 Python 文件；Gems 证据是原 correctness pytest、benchmark 和必要测试 helper。源码导出不执行 pytest、候选或设备探针；不等于执行验证。

审核使用现有 ArtifactReviewerAgent 的 `kind=tests`，不新建第二套 Runtime。报告保留已读文件、问题及缺失依赖。确定的 P0 测试契约错误（reference、候选注入、ABI、实际断言或计时对象错误）与 P0 转换错误阻断；一般覆盖不足作为建议。缺少必要依赖、无法读取完整证据也阻断，不能靠猜测通过。抽取阶段的 `kind=catalog` 保留原转换保真策略，代码审核 `kind=code` 保留 advisory 策略。

显式跳过生成 `review.json`，标记 `SKIPPED_EXPLICIT`，不会伪装成 `ACCEPTED`。通过审核或显式跳过后，Native 仍执行 reference Preflight/Eval；Gems 默认通过 KGS 执行原 benchmark 的 `--reference-only`（默认 level 为 `core`，无需显式传入），成功后才进入优化。不修改或逐个适配 correctness pytest，其 reference 仍在生成候选后的完整正确性测试中验证。测试源码审核或性能 baseline 可执行，都不代表完整正确性或性能已合格。

Gems 的执行结果保存在 `benchmark-reference.json`，标记 `validation_scope=benchmark_core`。KGS 校验原 benchmark fingerprint 和 core case 覆盖，返回 `ALL_SKIP`、`UNSUPPORTED`、超时或失败时不开始生成，不能当作验证通过；不会因此修改 source pytest 或减少 dtype/workload。Reference 调用复用 KGS 单卡 slot、隔离进程、超时强探针和 operation cancellation，KG 持久登记 `reference` operation，不伪造 candidate，不把执行结果写入优化 ledger。`skip_review` 也不跳过该调用。

KGS 导出有文件数/总字节上限，拒绝越界和链接，不静默截断。Gems 审核目前覆盖原 correctness/benchmark suite、测试 helper、benchmark case/shape 配置及 Profile hook；不导出任意路径、数据归档或所有第三方库。Reviewer 必须报告确实缺失的必要依赖；旧 KGS 没有源码导出能力时明确失败，用户需更新配套 KGS 或显式 skip，不自动降级为来源检查。

## 恢复边界

同一新任务的 resume 继续使用已经完成且 artifact 校验通过的 review_tests 回执，不重复调用已完成的步骤；这是恢复已完成工作，不是跳过未做的审核。新任务不复用外部抽取审核记录。审核失败的 attempt 保留，resume 会创建新 attempt。

旧 `review_catalog` 计划不自动迁移：旧任务用原 KG/KGS 快照查看和续跑，新计划使用新 workspace。不能重命名旧 scope 或把旧 `SKIPPED_TRUSTED_INSTALLED` 回执当成新审核通过。skip 选项和内部 `test_validation` 策略标识属于冻结计划，不能在续跑时静默改变；此前只有源码审核、没有 benchmark core 执行证据的实验也不能冒充新策略已通过。

## 验证边界

早期 host/API 验证使用 Gems `ff1befbd` / KGS `47d2ee2`；随后 Gems 将 reference-only 默认 level 设为 core 并合入 master #6681，再同步到 `kernelgen-dev`。这些旧组合仅是实现过程中的快照，不作为当前部署配套版本。

当前 KG 的 [KGS 锁定清单](../../../deployment/kgs.lock.yaml) 和 KGS `compatibility.yaml` 分别确定 Server exact commit 与 Gems 来源策略。Gems `kernelgen-dev@4772d816` / KGS `ba05292` 的真实 `relu6_` 验收结果见 [A100 验证记录](../../validation/gems_definition_review_a100_20260925.md)：其中区分了 host 回归、真实模型审核、core reference、候选评测、Profile 和最终代码审核；不能把单项通过等同于全部平台通过。
