# 优化实验中的 Workflow 边界

## 决策

KG 提供可独立调用和组合的 Workflow，不新增常驻编排器、共享任务队列或生产者—消费者框架。后续 Web 自行决定任务清单、跨任务去重、依赖触发和多机提交；现有 KG Coder lease、workspace 独占、取消与 KGS device slot 不迁移给 Web。

## 业务流程

NV 一阶段按需生成 pytest，审核、修正并在 NV 验证，之后通过 Gems adapter 运行 SimpleOpt。生成 pytest 是独立输入生产能力，可以由外层一次提交串联，但不放进 CatalogOptimize 的 prepare。

NV 一阶段未达标的算子可以直接进入 Native 二阶段。已有实现的多机复测产生按“算子 × 芯片”记录的 `kernel_todo_v3`；各芯片先使用 Gems adapter 再跑 SimpleOpt，称为特化。特化后仍未达标的任务才进入 Native 二阶段，不在多机特化前统一强制抽取。

Native 二阶段独立调用 CatalogExtractWorkflow，对源码和确认后的 pytest 进行抽取、review/修正，产出共享 Catalog，再由各芯片分别调用 CatalogOptimize 的 KernelGen 模式。同一来源和测试版本的一份已完成 Catalog 可以立即被任意目标消费，无需等待整批抽取完成。目标验证失败不允许目标优化器自行重抽或修改共享 Catalog。去重和何时触发属于调用方职责，不在 Workflow 内保存跨任务注册表。

历史各芯片 Triton 可以作为各自的 seed；只读 reference_code 可以是 CUDA、Ascend C 或 Triton。它们不替代 Catalog/Gems 的正确性与性能 reference，不因种子不同生成不同测试契约。

对一份已生成的 Gems 候选做可移植性验证时，独立的 [MultipleDeviceTest](multiple_device_test.md) 消费冻结的 Definition 与候选；每个目标先执行原 benchmark 的 core reference-only，只有通过后才执行该目标的候选 Preflight/Eval。它记录跨目标测试结果，不负责新代码生成、目标特化或下一轮调度。

## 当前实现与后续接入

- 已有：独立 CatalogExtract、独立 CatalogOptimize；不再提供 ExtractAndOptimize 组合入口；抽取结果通过普通路径交付，调用方不传 Bundle SHA256。
- 当前优化入口为 OperatorOptimize 的 `prepare_catalog → review_tests → optimize → code_review`；上传与来源校验归 prepare，所有来源默认审核测试契约，仅显式 skip 跳过。Native 目标验证保留在审核后，抽取不进入优化流程，也不新增自动调度。详见 [统一审核](review_tests.md)。
- 当前 Gems review 在源码审核后默认执行原 benchmark 的 core reference-only，复用 KGS 调度。尚未接入独立 pytest 准备 Workflow 或生成前 correctness reference-only；性能 baseline 可运行不表示原生 correctness pytest 已通过。
- TestWriter 当前会使用本机解释器验证并可能改写传入 Gems checkout。接入前必须隔离产物写入，明确 NV 执行环境与受调度的验证路径，保留真实执行/skip/注入证据；不能直接包装它便宣称支持 CPU KG + 远端 NV，也不能用 adapter Eval 冒充原生 pytest 验证。
- 已接入：`kg run --mode simple_opt|kernelgen` 统一调用 CatalogOptimize，支持 CLI 后台和 YAML Batch；模式只选择内部优化器。

每次调用都有明确输入输出和 workspace，细粒度运行状态来自现有 run control，性能事实来自 ledger。已完成的准备/审核可以按原产物校验复用，只有 optimize 恢复内部轮次；不增加另一套 round、Catalog 版本或任务状态真源。
