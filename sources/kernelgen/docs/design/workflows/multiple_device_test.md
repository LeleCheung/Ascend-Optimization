# MultipleDeviceTest：一份 Gems pytest 与候选的跨芯片复测

`MultipleDeviceTestWorkflow` 消费已导出的 v6.0 Gems Definition Catalog 和一份生成后的候选代码，在显式列出的 KGS 目标上依次复测。同一候选和 Definition Bundle 在工作区开始时冻结；Workflow 不生成新代码、不改写原 pytest、不裁剪 dtype 或 case，也不把某个目标的结果当作其他目标的硬件能力。典型输入是 `kg definition` 的 `catalog_path` 与成功 `kg run` 的 `.best_kernel.py`。

## 执行顺序

每台目标使用本地 loopback KGS 地址；远端目标通过 KG 的 SSH stdio HTTP proxy 到达，不接收远端直连 URL。Workflow 先读目标 `/status.api_version`、capabilities 和 scheduler；具备 Gems Bundle 绑定与 core reference 能力且有健康 slot 才上传同一内容寻址 Bundle。`/inspect` 核对目标 checkout 中的原始 pytest/benchmark 与冻结 Definition。随后立即调用候选无关的 `/reference`，保存 `flaggems.reference/v1` 原报告及逐 case dtype、失败阶段、异常类型和原文。`FAILED`、`ALL_SKIP`、`UNSUPPORTED`、超时或设备错误都结束该目标的测试；即使 KGS 总状态为 `PASSED`，只要 core 报告包含跳过的记录，该目标也不进入候选 Preflight/Eval。这样不会把某个 dtype 的 skip 误记为已通过；下一台目标继续。

只有 reference `PASSED`，才用同一份候选依次调用 Preflight 和完整 Eval，并保存原始 KGS 响应、每个 workload 结果、geo mean、目标状态和后置 scheduler。Eval 的 `PASSED` 还需有效计时、非 hack、非全跳过及全部 workload 通过才能把目标标为 `PASSED`。一台目标不支持 API/dtype、Gems 来源版本不匹配、编译/数值失败或 KGS 暂不可用，均在自己的报告中保留阶段和原因；不会通过删除 case 或改写 reference 变成成功。KGS 的请求线程与 device slot 仍由服务端分别管理，Workflow 默认串行访问目标。

## Python 调用

先用 `kg server status <name> --json` 确认每个命名实例的 `state=RUNNING`、`server.api_version=v6.2`、`config.flaggems_commit` 匹配 Definition 的 `source_revision`、scheduler 健康，并取其本地 `url`。例如：

```python
from kernelgen.workflows.multiple_device_test import MultipleDeviceTestWorkflow

result = MultipleDeviceTestWorkflow(cwd="runs/relu6-cross-device").run({
    "operator": "relu6_",
    "catalog_path": "runs/relu6-definition/catalog",
    "candidate_path": "runs/relu6-optimize/stages/optimize/work/1R/agent0/.best_kernel.py",
    "targets": [
        {"name": "ascend", "server_url": "http://127.0.0.1:24306"},
        {"name": "hygon", "server_url": "http://127.0.0.1:24302"},
    ],
})
print(result.model_dump_json(indent=2))
```

返回值是 `WorkflowResult[MultipleDeviceTestOutput]`，根目录的 `workflow_result.json` 保存完全相同的 JSON：顶层 `state` 为 `SUCCEEDED`、`FAILED` 或 `CANCELLED`，`output.results` 按输入顺序列出每台目标的结果，`output.passed_targets` 和 `output.completed_targets` 可直接供后续编排解析。某台失败不会阻止测试下一台；最终只要有目标失败，顶层 `state` 就是 `FAILED`。取消时返回 `CANCELLED` 和已完成目标的结果，不把未执行目标伪装成失败。

每台目标的 `targets/<name>/` 独立保存 `status-before.json`、`bundle.json`、`inspect.json`、`reference.json`、按已执行阶段保存的 `preflight.json` / `evaluate.json`、`status-after.json` 和 `result.json`。根目录 `plan.json` 绑定来源 commit、候选 SHA-256、Bundle ID、目标和评测设置；最终 `workflow_result.json` 是跨目标解析入口，不替代逐 case 原报告。运行中通过 `WorkspaceRunControl(workspace).progress()` 读取任务/目标 scope，事件在 `.kernelgen/run-events.jsonl`。首次运行必须使用新 workspace；第一版不自动续跑或并行调度历史结果。

本 Workflow 测试的是同一份已生成代码的可移植性，不保证针对各芯片最优。跨芯片计时不能用共用卡的结果声称独占性能；若要特化，应为失败或低收益的目标启动独立优化任务，仍使用其 KGS/pytest 原始契约和自己的 ledger。现场连接、容器及授权卡以 [机器清单](../../../tests/hosts.md) 为准，历史实验端口不能当作当前服务存活证明。
