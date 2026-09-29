# Square 算子第二次运行：2026-09-29

910B 工作区：
`/data/hanle/ascend-optimization/runtime/kg-controller/runs/square-private-2`。

本次在 `npu:0` 上对每组用例计时三次。Catalog 审核阶段通过 36/36 条用例，其参照检查返回的几何平均加速比为 1.0262。优化阶段的 Coder 未完成正式候选轮次。探索性性能分析和调试扫描触发了 KGS 的 32 个归档文件上限，随后仍继续扩展。应用户要求收尾时，已取消本次运行。

KG 最终状态为 `CANCELLED`：两个阶段完成，一个阶段取消。Worker 和本次调试子进程已退出；KGS 服务仍在运行。本次没有经过验证的候选加速比，也没有完整工作流的最终结果。

原始日志仍在隔离的服务器工作区。后续请按 `experiments/ascend910b/square/PITFALLS.md` 中的交接说明，使用新的工作区继续。
