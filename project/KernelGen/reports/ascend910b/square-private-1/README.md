# Square 算子最小验证：2026-09-29

环境为 KG 6.7.0、KGS 6.5.0、`tle_yy` 容器、NPU 0，以及使用 `deepseek-flash` 的 Claude 运行时。工作区：
`/data/hanle/ascend-optimization/runtime/kg-controller/runs/square-private-1`。

生成的 Triton 候选代码已在 Ascend910B4-1 上编译并运行。搜索阶段评测和独立最终复验均通过 27 条正确性用例及 9 条计时用例，正确性误差为零。27 条正确性用例实际上是 9 种形状与数据类型组合重复三次：形状为 `[2,3]`、`[128,256]`、`[1024,1024]`，数据类型为 fp32、fp16、bf16。参照实现是目标 NPU 上的 `torch.ops.aten.square(x)`。

搜索阶段的几何平均加速比为 1.04505254，最差为 0.24341016。最终复验测得几何平均加速比 1.15224995，但 KG 因 `NEEDS_RETEST / TIMING_DRIFT` 拒绝确认最终结果：一条参照实现的延迟从 0.00604846 ms 变为 0.02638274 ms，变化 4.36 倍。本次运行没有进入代码审核阶段。这些数据证明设备评测链路已运行，不能作为稳定加速比或完整工作流成功的证据。

预算为一轮、一次 Coder 会话、不提供性能分析反馈、每组计时一次。KGS 的设备计时方式仍为 `profiler`；`--no-profile` 只关闭优化过程中的性能分析反馈，不关闭设备计时。软件版本：flagtree `0.6.0+ascend.gitc286cba6`、CANN 9.0.0、驱动 25.2.0。

本目录归档了 `run-request.json`、`round-0001.json`、`output.json` 和 `ledger.json`。后续新运行改为每组计时三次。原始日志及最终复验记录仍保留在服务器工作区。
