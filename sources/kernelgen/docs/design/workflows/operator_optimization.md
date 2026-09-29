# 算子优化入口与共享引擎

`OperatorOptimizeWorkflow` 是单算子、单目标环境的统一业务入口，接受 KGS 已安装的 Gems adapter / Native Catalog、本地 Native Catalog 或由 [GemsAdapterDefinitionWorkflow](gems_adapter_definition.md) 导出的本地 Gems adapter Catalog。它不抽取 Native Catalog，也不生成 pytest，不承担跨机器常驻调度和 PR 操作。

## 默认行为

未指定参数时采用 Gems adapter、`kernelgen`、1 个并行 Coder、1 个 epoch、每个 Coder 每个 epoch 最多 10 rounds；启用 KernelGen 原有 Profile 默认行为。显式 `--mode simple_opt` 继续使用单 Coder 引擎，不引入第二套优化循环。CLI、Python Workflow 和 Batch 从同一组无依赖默认常量取值；显式参数与已保存请求不被新默认值覆盖。

```bash
kg run --definition negative --eval-server http://127.0.0.1:19808
```

目标硬件仍来自 KGS `/status`，不是本地 Torch。未显式选择 Catalog 时使用 `flaggems-adapter-definitions`；选择 `catalog_path` 时不再隐式补入默认 `catalog_name`。公共 Python 输入可只提供 `operator`，优化输入的 definition_name 由它补齐；若两者显式冲突仍拒绝执行。

## 代码组织

```text
workflows/optimization/
  workflow.py        OperatorOptimizeWorkflow
  inputs.py          统一入口的模式配置
  options.py         共享的单 Coder 选项，不依赖旧 SimpleOptWorkflow
  operations.py      输入准备、验收、优化调用、代码审核
  sources.py         KGS 契约与本地输入快照
  knowledge.py       单 Coder Knowledge 准备
  single_coder/      SingleCoderOptimizationWorkflow
  kernelgen/         KernelGenWorkflow
```

KernelGen 内部继续复用 SingleCoderOptimizationWorkflow。新任务只使用 OperatorOptimizeWorkflow，单算子与 YAML Batch 均由 `kg run` 提交。旧 SimpleOptWorkflow 与 Python Batch 实现移入 `workflows/legacy/`，仅服务历史目录的续跑和结果读取；不再保留 `workflows/simple_opt` 和 `workflows/batch_simple_opt_definition.py` 公共入口。废弃的 `workflows.catalog_optimize`、`workflows.kernel_gen` 和 `workflows.kernel_optimization` 导入别名也已删除。

日常提交、状态查询、取消和续跑统一使用 `kg run/status/cancel/resume`。`examples/catalog_optimize` 的独立 JSON launcher 已删除；Runtime 初始化由 `cli/optimization.py` 承担，`cli/runner.py` 直接传递验证后的请求，不重建 argv。SimpleOpt 单算子及 Batch 示例仅转交 CLI 参数，默认后台提交，单任务可用 `--foreground`；Batch 使用 `--batch-file`，不再接受旧的重复 `--definition-name` 参数。v1 请求由 `cli/legacy_simple_opt.py` 续跑；旧 Python Batch 使用 `python3 -m kernelgen.cli.legacy_batch_simple_opt` 和原参数。直接嵌入 Python 程序时使用 `OperatorOptimizeWorkflow`，不再依赖 examples 作为生产接口。

## 编排与恢复边界

新计划执行 `prepare_catalog → review_tests → optimize → code_review`。所有来源默认审核测试契约，仅显式 skip 跳过，见 [统一审核](review_tests.md)。原 `review_catalog` 计划使用原版本续跑，不改写旧 scope 或把旧免审回执迁移为已审核；新任务使用新 workspace。优化结果仍由 ledger 和独立最终复验决定，Profile 和 advisory code review 不被误当成新的性能真源。

优化阶段仍在稳定的 `stages/optimize/work` 目录中恢复。stage attempt 只记录调用回执，不代替 Coder 轮次或 epoch 状态；已完成阶段按原恢复规则复用。reference/seed 文件摘要、目标环境快照、协作式取消与本地 Coder lease 权重保持原语义。

## 后续独立配套

默认 Gems 开发来源为官方 `kernelgen-dev`；[Definition Workflow](gems_adapter_definition.md) 与此入口串联验证。不得把旧 KGS 内置 Definition 的运行结果当作新生成 Definition 已被消费的证据，也不得将 Gems pytest 翻译成 Native workload 来绕过原始测试。最终验收必须记录实际 Gems commit、生成 Definition 的摘要、KGS 执行绑定、ledger 和真实优化输出。

H20 验证通过后再独立开发 `MultipleDeviceTestWorkflow`：把 NV 已验证的同一份候选代码与 Definition/Catalog 交给指定的其他目标 KGS，执行 Preflight、正确性和 Benchmark，记录各目标自己的 baseline、候选耗时和加速比。它不生成或修改代码，不自动调用特化；输出达标、性能未达标、正确性失败和环境/能力阻塞的分类及后续特化输入。当前实验默认阈值为 0.8，须显式记录且可配置，不把 Workflow 成功等同于性能达标。各目标独立结果和恢复记录统一接入 run control，单机失败不阻塞其余目标；该功能尚未实现。
