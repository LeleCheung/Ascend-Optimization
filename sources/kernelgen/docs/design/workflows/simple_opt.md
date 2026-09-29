# SimpleOpt：统一入口与历史恢复

SimpleOpt 是 OperatorOptimizeWorkflow 的一种优化模式，不再作为独立业务 Workflow。新任务统一使用 `kg run --mode simple_opt`，批量使用 `kg run --mode simple_opt --batch-file batch.yaml`；Python 集成使用 `workflows.optimization.OperatorOptimizeWorkflow`。共同调用链为 OperatorOptimize 准备/验证输入，然后直接运行 `optimization.single_coder.SingleCoderOptimizationWorkflow`，最后记录代码审核。单 Coder 引擎仍负责轮次、Profile、最终复测和 distillation，不新增优化循环。

`examples/simple_opt/run_example.py` 和 `examples/batch_simple_opt_definition/run_example.py` 仅把参数交给同一个 CLI parser 和提交实现。默认后台执行，单任务可显式 `--foreground`；新 Batch 每算子一个独立 RunRequest、进程和 workspace，根目录仅保存子任务索引，不再使用 Python Batch 的第二套汇总文件。

## 历史边界

删除原 `workflows/simple_opt`、`workflows/batch_simple_opt_definition.py` 的公开导入，旧实现隔离在 `workflows/legacy/`。这不是把旧状态转换为新状态：已有 flat workspace 的 ledger、最终输出、取消 generation 和恢复语义均保持原样。

- CLI v1 请求：`kg resume` 仍经 `cli/legacy_simple_opt.py` 调用 legacy SimpleOpt，再复用相同的单 Coder 引擎。
- 旧 Python Batch：使用 `python3 -m kernelgen.cli.legacy_batch_simple_opt` 和原参数，保留 `definitions/<operator>/.ledger.json` 等位置；历史 monitor 和 PR 结果读取改用 legacy 模块，不改写文件格式。不要使用新示例入口重解释旧 campaign。
- 新任务：通过 OperatorOptimize 在 `stages/optimize/work` 保存优化结果；Batch 各子 workspace 也保持相同布局。不要向单 Coder 引擎添加本地 Catalog 猜测或 Batch 调度以兼容旧入口。

共享选项仍位于 `optimization/options.py`；legacy 仅引用它，不复制字段或默认值。历史恢复回归继续保留；新入口测试覆盖 v2 请求、单 Coder 权重、独立 Batch 索引、退出码和 CLI 参数透传。当前完整职责见 [统一优化入口](operator_optimization.md)。
