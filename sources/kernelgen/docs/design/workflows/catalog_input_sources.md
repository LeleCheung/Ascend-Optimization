# OperatorOptimize 输入来源

`kg run --mode simple_opt|kernelgen` 和 Python launcher 统一经过 `OperatorOptimizeWorkflow` 准备来源、冻结评测快照，再选择优化器。Gems adapter 也支持 KernelGen，直接使用 KGS 上原 pytest 的输入、reference 和计时，不要求抽取 Native Catalog。

## 用户只选择名称或目录

`catalog_name` 和 `catalog_path` 必须且只能提供一个，放在 OperatorOptimize 请求顶层。`optimization.catalog_name` 不接受第二次来源选择。用户不传 `bundle_id`，也不需要计算 SHA256：本地 Native 目录由 KG 打包，SDK 按内容地址查询，缺失才上传；上传不等于安装到 KGS `data/`。

安装式 Native 示例：

```json
{
  "catalog_name": "kernelgenbench",
  "operator": "kernelgenbench_square",
  "optimization": {
    "definition_name": "kernelgenbench_square",
    "mode": "kernelgen",
    "eval_server_url": "http://127.0.0.1:8000"
  }
}
```

本地目录将 `catalog_name` 替换为 `catalog_path` 即可；支持 Native v6.2，以及 [GemsAdapterDefinitionWorkflow](gems_adapter_definition.md) 导出的纯 v6.0 Gems Definition Catalog，后者必须携带冻结来源记录，不能任意复制一个内置 adapter Catalog 冒充导出结果。安装式 Gems adapter 将来源名称换成对应 Catalog，`mode` 可选 `simple_opt` 或 `kernelgen`。公共请求仍要求 `operator` 与 `optimization.definition_name` 一致。

推荐使用 `kg run --definition negative`（默认 Gems adapter 和 KernelGen）；Python 集成直接使用 `kernelgen.workflows.optimization.OperatorOptimizeWorkflow`，不再提供独立 Catalog example launcher。远程目标 URL 必须是本地 SSH stdio proxy 地址。KernelGen 的 `n_parallel` 仍占对应数量的 Coder lease，需在原 WorkerLeasePool 中配置足够容量；不改变 KGS 请求线程和设备 slot。

## 真源与执行边界

安装式输入要求目标 `/status.capabilities.operator_contract.enabled=true`，通过 KGS `/operator-contract` 读取 Definition、reference 与声明的 correctness/timing workloads，再通过 `/inspect` 冻结 evaluator kind 和 benchmark fingerprint。该接口来自 KGS !55；部署前必须确认配套 checkout 和客户端具备这项能力，不能仅凭 release 名称推断。缺失能力明确失败，不回退到本地 Catalog。安装式输入不要求 Bundle 上传 capability。

所有输入统一执行 `prepare_catalog → review_tests → optimize → code_review`。prepare 建立绑定、保存评测快照和审核源码；本地输入在这里上传，不再增加 distribute。review_tests 默认审核 Native 测试契约或原 Gems pytest/benchmark，不因已安装来源或历史审核记录自动免审；只有显式 `skip_review=true` 记录为 `SKIPPED_EXPLICIT`。审核通过或显式跳过后，Native 仍执行 reference Preflight/Eval，Gems 执行原 benchmark 的 `--reference-only`，使用其默认 core level，成功后才进入优化。Gems correctness reference 仍留在候选完整正确性测试中验证，不能把性能 baseline 能跑当成 correctness 已通过。详情见 [统一测试契约审核](review_tests.md)。

本地导出的 Gems Definition 额外要求 Bundle capability 的 `evaluators` 包含 `flaggems`。KGS inspect 核对原始 suite、文件摘要和源码 commit，但不因此跳过 review_tests。候选 Preflight/Eval/Profile 仍使用目标 Gems 的原始 pytest；更新 Gems 后必须重新导出匹配的 Definition，不能复用旧 Bundle 绕过来源一致性校验。

快照新增 `evaluator_kind`，旧快照默认为 native。带已核验 fingerprint 的快照直接供评测适配器使用；安装式来源每次构造评测请求时核对目标导出的契约及 inspect fingerprint，不再读取本地目录。没有 fingerprint 的旧本地快照保留原校验路径，新安装式来源不会降级到这条路径。任何契约、显式 null 默认值或 fingerprint 漂移都不能被相同 Catalog 名称掩盖。

准备步骤持有一份冻结快照，下游通过路径引用，不在准备和 readiness 各复制一份可修改契约。优化器原有工作区冻结机制仍保留，用于独立 Coder 和恢复；已有服务端快照不能被内容不同的新请求覆盖。完整 SHA256 只在内部快照和审计证据中使用，不增加短 ID 注册表或用户别名映射。

## 恢复与验收范围

新计划中，已完成的 prepare 和 review 通过原 receipt/artifact 校验跳过，优化继续使用稳定的 `stages/optimize/work`。review 失败或取消后在新 attempt 重跑审核与验证，复用 prepare 快照，不重复上传；优化恢复内部 ledger。恢复不改变绑定类型，不重新查询并覆盖原快照。恢复后发生新的评测时仍核对目标契约。已经全部完成的任务只读取历史证据，不声称重新验证了当前服务器。旧的含 `distribute_catalog` 计划由原版本续跑，本次不增加旧计划转换器。

Host 测试覆盖 Native/Gems × SimpleOpt/KernelGen 四种安装式组合、单一来源选择、远端能力缺失、reference null 语义、契约与 fingerprint 漂移、已完成任务恢复、原 Bundle 流程和取消登记。Gems KernelGen 的冻结与恢复继续保留 evaluator kind 和 fingerprint，不补造 Native workloads。配套 HTTP 测试使用真实 KGS 路由、SDK 与 Schema，但模拟设备执行；这些结果不能替代真实 CLI 后台、模型生成和多芯片 E2E 验收。

四步 prepare/review 重构后的 A100 真机证据见 [本地 Native Catalog 上传 E2E](../../validation/catalog_prepare_review_a100_20260914.md)：SimpleOpt、KernelGen 均使用真实模型、审核和评测；该次验证是 Python launcher，不是统一 CLI 入口或新 pytest 生成流程。
