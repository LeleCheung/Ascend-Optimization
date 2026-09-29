# 共享优化器输入构造

本次小重构在 `workflows/optimization/single_coder/inputs.py` 提供 `build_optimizer_input()`，SimpleOpt 与 KernelGen 共用公共参数转交逻辑，不改变优化器、CLI 参数或持久化格式。

函数只转交源模型与现有 `SingleCoderOptimizationInput` 同名的字段，再应用调用方准备的上下文；不复制字段表、输入 Schema 或默认值。源模型没有提供的字段继续由优化器采用原有默认值。显式 Eval timeout 通过 `TimeoutPolicy` 派生 transport budget，KernelGen 工作流的 `timeout` 不映射为 Eval timeout。最终校验仍由优化器的输入模型负责。

调用方继续负责 Catalog 加载、快照身份校验、DPS 策略、reference/Knowledge 准备和 analysis/seed 分配。SimpleOpt 的默认 DPS 与 KernelGen 的推断逻辑保持各自原有语义。共享函数不读写文件、不请求 KGS、不改变调用方对象，也不创建或解析 Catalog。

仍在独立开发的 Catalog 优化 Workflow 待公共改动合入后接入：将已验证的 Bundle 快照和优化配置交给同一函数，不重新解析内置 Catalog。本分支不包含该业务 Workflow，也不混入 SimpleOpt 拆包、Workflow 更名或 worker lease 迁移。

## 共享 Analyzer 的评测上下文

KernelGen 的共享 Analyzer 与 Coder 通过 `workflows/optimization/kernelgen/epoch.py::resolve_optimization_context()` 解析相同的 Definition 和 workload。存在冻结的 Native/Bundle 快照时优先使用快照并校验身份，不重新加载本地 Catalog。Analyzer prompt 与 Coder 共用 `agents/_workload_prompt.py` 的确定性分 phase 摘要，最多展示 10 项并明确未展示的用例仍须全部验证；这只是提示词预算，不缩减实际评测覆盖或新增另一份评测事实。

`preparation.resolve_server_target()` 从同一次已校验的 KGS `/status` 响应中保留 api_version、backend、timing、target、software 和 metadata，供共享分析使用；不注入 scheduler 或 Debug 私有路径。缺失元数据表示未知，Analyzer 不得用本机 Torch/Triton 或猜测来填补目标设备事实。有效的 resume analysis checkpoint 继续复用，不因补充上下文而重新生成；最终收尾路径继续沿用原目标校验。创建分析 workspace 时保留 `cross_epoch_knowledge` 对旧知识继承的控制，使知识消融开关与这项输入补充互不覆盖。

回归入口为 `tests/test_analyzer_evaluation_context.py`，覆盖冻结 workload 一致性、单次 Server 状态读取、checkpoint 复用、缺失事实提示、身份不匹配拒绝、摘要不变性，以及知识继承开关的两种取值。

## 验证

新增 `tests/test_optimizer_inputs.py` 固定 SimpleOpt、KernelGen 重构前的完整输入映射，覆盖默认值与显式覆盖、DPS、轮次/session 上限、评测预算、Profile/Knowledge、reference、analysis 和 seed；另验证传入的 Bundle 上下文不触发 Catalog 加载。共享函数测试不代替真实 Bundle 分发或执行验证。

相关 host 回归范围：

```bash
python3 -m pytest -q tests/test_optimizer_inputs.py tests/test_simple_opt.py tests/test_kernel_gen.py tests/test_retest.py tests/test_batch_simple_opt_definition.py tests/test_evaluation_snapshot.py tests/test_timeout_policy.py tests/test_run_options.py
```

独立 worktree 测试必须先核对 KG/KGS 的 `__file__`，KGS 使用 KG 锁定的 commit。此次改动不安装依赖、不启动模型或设备服务；host 回归不能代替真实芯片 E2E。
