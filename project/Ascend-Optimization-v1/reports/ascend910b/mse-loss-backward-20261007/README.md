# `mse_loss_backward`：Ascend 910B 阶段报告

## KG 无 profiler

正式运行通过 42/42 workload，几何平均加速比 1.326x，最差 1.064x。候选为融合的 Triton elementwise kernel，按输入规模选择 block，并保留非连续输入 fallback。

## profiler 版本

正式运行：`mse-loss-backward-profile-20261007`。结果为 42/42 正确、42/42 计时，几何平均加速比 1.243x，最差 0.937x。说明 profiler 流程下总体略快，但最慢 workload 仍低于 PyTorch。

## 复现证据

- 无 profiler：`runtime/kg-controller/runs/mse-loss-backward-noprofile-63c4721c`
- profiler：`runtime/kg-controller/runs/mse-loss-backward-profile-20261007`
- profiler 正式结果：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`
- profiler 运行请求启用了 `--profile`，设备为 Ascend 910B。
