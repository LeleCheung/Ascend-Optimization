# mse_loss_backward 优化报告

## 结论

在 Ascend 910B 上，KG 无 profiler 正式闭环通过。候选在 `42/42` 个 workload 上通过正确性和计时评测，几何平均加速比为 `1.326x`，最差 workload 为 `1.064x`，候选状态为 `KEEP`。

## 方法

连续同形状输入使用融合的 Triton elementwise kernel，在 host 端折叠 `2/numel`，按输入规模选择 block，并缓存已验证的 compiled-kernel launcher。广播、非连续、标量梯度和任意 rank 输入走通用 fallback。

## 证据

- 910B workspace：`runtime/kg-controller/runs/mse-loss-backward-noprofile-63c4721c/`
- Definition revision：`63c4721c5febf2ff563e6ea3627617fb809abc2e`
- 正确性：27/27
- 计时：15/15
- 阶段状态：`SUCCEEDED / COMPLETED`

## 限制

本报告是无 profiler 结果；尚未完成该算子的 profiler 运行及独立 profiling 改进版。
