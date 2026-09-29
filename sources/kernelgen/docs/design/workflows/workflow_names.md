# Workflow 源码命名与兼容边界

## 当前组织：统一优化入口与执行引擎

2026-09-24 的源码组织调整将统一入口改为 `OperatorOptimizeWorkflow`，实现位于 `workflows/optimization/workflow.py`；单 Coder 和 KernelGen 分别位于 `optimization/single_coder/` 与 `optimization/kernelgen/`。单 Coder 输入输出名称明确为 `SingleCoderOptimizationInput/Output`。共享选项与 Knowledge 准备不再依赖旧 SimpleOpt 包装层。详见 [统一优化入口](operator_optimization.md)。

旧 package-level 导入别名与 `examples/catalog_optimize` launcher 已删除，调用方使用新模块，日常优化统一使用 `kg run`。持久化 Workflow.name、stage scope、ledger 与结果文件名不随 Python 名字变化。新任务默认使用 Gems adapter、KernelGen、1 Coder、1 epoch、最多 10 rounds；显式参数与旧任务请求仍保留原值。以下各节是此前改名的历史记录，不代表当前目录仍保持那些布局，也不将旧测试结果冒充本次验证。

## 先前单 Coder 名称调整（历史）

`KernelOptimizationWorkflow` 更名为 `SingleCoderOptimizationWorkflow`，明确它是 SimpleOpt 和 KernelGen 共用的单 Coder 执行层，不是 Catalog 准备、审核、上传与优化的全流程入口。仓内调用方统一更新，不保留旧类名别名。导入路径仍为 `kernelgen.workflows.kernel_optimization`；`KernelOptimizationInput/Output`、目录名、持久化 `name="optimize_definition"`、输出文件、ledger 和取消/续跑协议均不变。本次仅改源码类名，不发布版本或移动 tag。

本次 host 对照：改名前 `origin/dev@124a4cf8` 等价文件树为 192 passed / 14 failed；改名后为 193 passed / 同样 14 failed，新增公开导出与共享调用方一致性测试通过。失败均在 `test_coder_workflow_state.py`，与基线逐项一致，未混入修复。测试范围为 `test_workflow_names`、`test_simple_opt`、`test_simple_opt_package`、`test_optimizer_inputs`、`test_kernel_gen`、`test_catalog_optimize`、`test_extract_and_optimize`、`test_retest`、`test_coder_workflow_state` 和 `test_cli_lifecycle`。两组测试分别核对 worktree 导入路径，KGS 均为 `00dfc869`；未启动模型或芯片实验。

## 先前命名迁移（历史）

按职责调整主线 Workflow 的业务名称及目录，不改变优化算法、阶段执行器或业务接入状态。前两项通过 !104 合入，公共 Catalog 抽取命名作为独立后续重构；Catalog 抽取加优化的业务编排仍在独立分支，待公共变更合入后接入。

| 原名称与目录 | 新名称与目录 | 职责 |
|---|---|---|
| `OptimizeDefinitionWorkflow`，`workflows/optimize_definition/` | 当时为 `KernelOptimizationWorkflow`，`workflows/kernel_optimization/`；当前类名见上节 | 准备好的单算子的单 Coder 优化执行，含 Profile、最终复测与 distillation；不修改 Definition |
| `OperatorLifecycleWorkflow`，`workflows/operator_lifecycle/` | `OperatorDevelopmentWorkflow`，`workflows/operator_development/` | 单算子的开发阶段编排与控制；当前仍是 dummy 业务执行，不因改名接入真实 Agent |
| `FlagGemsV62AgentExtractWorkflow`，`workflows/flaggems_v62_agent_extract.py` | `CatalogExtractWorkflow`，`workflows/catalog_extract.py` | 现有 FlagGems Native 抽取与 correctness 覆盖报告；不因改名扩大支持的源码或协议范围 |

输入输出类型对应改为 `KernelOptimizationInput/Output` 和 `OperatorDevelopmentInput/Output`，输出文件常量改为 `KERNEL_OPTIMIZATION_OUTPUT_FILENAME`。后续 [Workflow build 重构](../runtime/stage_orchestration.md) 用 `WorkflowContext`、`WorkflowResult` 替换了独立 Stage 接口，`run_campaign()` 保留。所有仓内 Python 调用方、测试与当前设计引用同步更新，旧目录不保留转发空壳；外部 Python 使用方需要更新 import、类型名称和生命周期返回值访问方式。该源码接口迁移应纳入后续软件发布说明，本分支不发布版本或改变 Protocol。

## 不变的协议与状态

- CLI 仍使用 `kg run --mode simple_opt|kernelgen|lifecycle`；现有 Python example 路径 `kernelgen.examples.operator_lifecycle.run_example` 继续可用。
- Workflow 的持久化 `name` 保留 `optimize_definition` / `operator_lifecycle`，既有日志前缀及事件类型不改名。
- 优化输出仍为 `optimize_definition_output.json`；ledger、最终复测及 best kernel 的事实归属不变。
- `operator-lifecycle.json`、阶段报告、campaign 索引的 `kind`、schema version、输入摘要、取消 generation、owner 检查和阶段路径保持原样；不迁移或重写旧 workspace。

源码类名不等于持久化标识。不要在后续清理时全局替换以上字符串，否则会破坏历史产物读取、完成报告复用或续跑计划匹配。SimpleOpt、KernelGen 的公开名称不在本次更名范围，运行控制层也不改名。

## 验证

本节记录 !104 的优化与算子开发命名验证；后续 Catalog 抽取重命名验证单列在文末。

`tests/test_workflow_names.py` 固定上述输出文件名、Workflow 标识、计划 Schema、campaign kind 和恢复语义；原 Workflow、CLI、ledger、汇总工具、Profile/最终复测等 host 回归同步使用新导入。

另外以更名前的 `4c08529c` 实际创建三组临时 workspace（完成、等待、取消），分别由旧代码与更名后的代码续跑对照：阶段调用序列相同，原有报告字节不变，取消 generation 正常递增，owner 释放。取消发生在完整阶段结果保存后时，该完成报告仍可复用，不能要求无条件重跑该阶段。此验证不启动真实模型、KGS 或设备，不替代芯片 E2E。

2026-09-11 验证记录：配套 KGS 为锁定的 `7f1110be26e1d3a37195dbf5fee96cc99ca516bb`，各次测试均核对 KG/KGS 导入路径。首轮相关回归 253 项通过；扩展到 ledger 和 Coder 历史夹具等测试后，更名前 `4c08529c` 为 282 passed / 16 failed，更名后为 287 passed / 16 failed。新增 5 项兼容测试全部通过，失败名单一致：`test_ledger_best_source.py` 的两个 winner 选择用例，以及 `test_coder_workflow_state.py` 的 14 个最终复测、Profile 和恢复相关用例。它们在本次重命名前已失败，需单独处理，不能将此结果描述为全部测试通过。

扩展回归命令（旧基线不包含 `test_workflow_names.py`，并使用旧名 `test_operator_lifecycle.py`）：

```bash
python3 -m pytest -q tests/test_workflow_names.py tests/test_simple_opt.py tests/test_simple_opt_package.py tests/test_optimizer_inputs.py tests/test_kernel_gen.py tests/test_operator_development.py tests/test_cli_lifecycle.py tests/test_workflow_lifecycle.py tests/test_retest.py tests/test_run_control.py tests/test_batch_simple_opt_definition.py tests/test_knowledge_distiller_reports.py tests/test_monitor_batch_simple_opt.py tests/test_summarize_batch_simple_opt_campaign.py tests/test_summarize_multi_device_batch.py tests/test_ledger_best_source.py tests/test_coder_workflow_state.py tests/test_cli_batch.py tests/test_evaluation_snapshot.py
```

## Catalog 抽取命名的后续重构

输入输出类型同步改为 `CatalogExtractInput/Output`，Python 使用方改为从 `kernelgen.workflows.catalog_extract` 导入；旧模块不保留转发壳。现有 `kernelgen.examples.flaggems_v62_extract.batch_extract` launcher 路径和参数不变，Workflow `name="flaggems_v62_agent_extract"`、错误消息、Batch 报告 Schema 与产物文件名不变，不改 CLI、运行状态或发布版本。

当前输入仍为 `operator`、`flaggems_repo`、`case_list_path`；Workflow 仍调用原 FlagGems Agent 和覆盖检查，返回 `operator`、`extraction`、`accuracy_coverage`，不直接写入共享 Catalog。源码准备、用例清单采集、统一落盘封装、硬件中立抽取及 PR URL 输入均未在本命名重构中实现；更通用的名称不是更广泛能力的声明。具体 Agent、oracle 协议和 FlagGems 来源适配保留原名称。

`tests/test_flaggems_v62_batch_extract.py` 固定旧输入输出字段、必填性、持久化标识与报告 Schema，验证 `bind()`、Runtime/workspace 传递和抽取不落盘；原 Batch 用例继续覆盖并行执行、调用方有序落盘、失败隔离、已存在包与用例缺口。配套 Agent、覆盖检查、持久化与 Native 抽取测试一并回归，host 模拟结果不代表硬件中立抽取或真实芯片 E2E 已通过。

2026-09-11：基于 `dev@03693da2`，使用锁定 KGS `7f1110be`，核对导入路径后以下 184 项全部通过；这组结果不覆盖或修复上节记录的 16 项历史失败。

```bash
python3 -m pytest -q tests/test_flaggems_v62_batch_extract.py tests/test_flaggems_v62_direct_agent.py tests/test_flaggems_v62_extract.py tests/test_flaggems_v62_metax_parity.py tests/test_flaggems_extractor.py tests/test_flaggems_batch_extract.py tests/test_workflow_names.py tests/test_cli_lifecycle.py
```

## ExtractAndOptimize 业务编排命名

本节保留历史命名及验证记录。该组合 Workflow 现已删除；当前分别使用 CatalogExtractWorkflow 和 CatalogOptimizeWorkflow，通过 Catalog 路径交接。

Catalog 业务分支已接入 !101–!105 的共享输入、Coder 租约、SimpleOpt 拆包与公共名称。`CatalogOptimizeWorkflow/Input` 改为 `ExtractAndOptimizeWorkflow/Input`，源码目录改为 `workflows/extract_and_optimize/`（现已删除），测试同步改为 `test_extract_and_optimize.py`。外部 Python import 需要更新，旧 Workflow 目录不保留转发壳。原 [Catalog 设计](catalog_optimize.md) 中的旧名是当时的设计和实验记录，当前入口以 [Catalog 设计](catalog_optimize.md) 为准。

现有 `kernelgen.examples.catalog_optimize.run_example` launcher 路径保持兼容，`run/status/cancel` 参数和退出码不变；不新增 kg mode。Workflow `name="catalog_optimize"`、五阶段名称及顺序、输入计划摘要、`operator-lifecycle.json`、receipt 和取消 generation 保持原样。优化仍在 `stages/optimize/work` 执行，使用已验证 Bundle 快照和一个 Coder 名额；ledger、最终复测、报告和代码审核门禁不变。

更名时抽取阶段改用公共 `CatalogExtractWorkflow`，Catalog 落盘和 Bundle 打包仍在外层。后续 [Catalog 流程拆分](catalog_optimize.md#三个独立入口) 将落盘移入 Extract，新增只接受已有 Catalog 的 `CatalogOptimizeWorkflow`，原组合复用其下游阶段。组合入口仍只支持单算子、单目标、完整阶段与 SimpleOpt，不直接接收 PR URL；独立 Extract 已接入 [Gems PR 与自动用例采集](flaggems_pr_extract.md)。硬件中立完整抽取、统一源码输入与多目标调度仍未完成。

更名前 `ac8557cf` 实际创建完成、审核等待、取消三类临时 dummy workspace，再使用新类续跑：完成阶段报告与计划字节不变，等待阶段正常产生下一 attempt，取消后复用已保存的完整阶段结果，generation 正常递增且 owner 释放。`tests/test_extract_and_optimize.py` 固定原持久化 name，并覆盖公共抽取接口到落盘、打包和摘要登记的调用路径。该验证不调用真实模型或设备，不替代此前昇腾 E2E。

2026-09-11 接入验证：配套 KGS 为业务分支依赖的 `a5749f5f`，已核对两个 worktree 的导入路径。以下 host 回归 310 项通过；原有 ledger/Coder 历史失败用例不在本组范围内。

```bash
python3 -m pytest -q tests/test_extract_and_optimize.py tests/test_flaggems_v62_batch_extract.py tests/test_flaggems_v62_direct_agent.py tests/test_flaggems_v62_extract.py tests/test_simple_opt.py tests/test_simple_opt_package.py tests/test_optimizer_inputs.py tests/test_worker_pool.py tests/test_operator_development.py tests/test_kernel_gen.py tests/test_cli_lifecycle.py tests/test_workflow_names.py
```
