# 删除组合入口的回归验证

## 范围

KG 分支 `refactor/remove-extract-and-optimize`，基于 `origin/dev@c742adc0e2a474b6785da055cd1f862493de4da3`，验证包含本分支未提交的删除与文档修改。配套 KGS 为锁定的 `49fcf4ecb86a2c3ec100b5194dba276100232b5e`：KG v6.3.1 开发快照 / KGS v6.3.4 / Protocol v6.2。本次没有发布或移动 tag，没有启动 KGS、模型或 GPU 任务，没有安装依赖。

独立 CatalogExtract 和 CatalogOptimize 保留；删除 ExtractAndOptimize、对应 CLI mode 和 Python launcher 选择器。原组合 workspace 不迁移，需原版本查看或续跑。独立优化的取消、receipt、ledger 和恢复语义不变。

## Host 回归

通过临时父目录的 `kernelgen` 符号链接隔离导入，已核对 KG 来自 `worktrees/kg-remove-extract-and-optimize`，KGS 来自 `worktrees/kgs-ppu-contract-validation`。设置 `PYTHONDONTWRITEBYTECODE=1` 和临时 `KERNELGEN_CLI_HOME`，不操作真实 CLI 状态。

```bash
python3 -m pytest -q tests/test_cli*.py tests/test_catalog*.py tests/test_run_options.py tests/test_kernel_gen.py tests/test_simple_opt.py tests/test_simple_opt_package.py tests/test_run_control.py tests/test_workflow*.py tests/test_worker_pool.py tests/test_flaggems_v62_batch_extract.py tests/test_progress_schema.py
```

结果：433 passed、1 skipped。跳过的是 `test_catalog_input_sources.py` 中需要 Torch 的测试，宿主机未安装 Torch；不能把它视为通过。专项 `test_install_kgs.py`、`test_cli_python_environment.py`、`test_cli_install_errors.py`、`test_cli_runtime.py` 为 53 passed，与前组存在重叠，不累加为独立测试总数。

## ONBOARDING 与历史结果

已检查修改文档的 90 个本地链接，无缺失目标。ONBOARDING 的 Server 启动、状态、优化提交、详细状态、日志跟随、原始日志、历史和停止共 8 组命令通过当前 CLI 解析；仅解析，不实际部署。安装脚本复用现有部署逻辑，通过上述 host 测试验证，不代表在全新系统重做安装验收。

ONBOARDING 去掉不必要的 `--target-hardware`，由 KGS 返回目标事实；补充 `kg status --detail`，明确独立抽取与统一优化入口。当前设计文档删除旧组合入口的可执行示例，历史报告保留原样。

使用本分支 CLI 只读查询此前真实实验 `kernelgen/runs/unified-catalog-cli-e2e-MYavP4` 的 6 个任务：安装式 Native SimpleOpt、安装式 Native KernelGen、Gems SimpleOpt、本地 Native Batch 两个子任务、Bundle Profile。均能读取 status/history；普通 prepare/review/code_review 的进度为空对象，根任务为 4/4，优化历史继续来自嵌套 ledger。此前真实设备运行证据见 [统一入口 A100 E2E](unified_catalog_cli_a100_20260914.md)，本次只读验证不是重新执行 E2E。
