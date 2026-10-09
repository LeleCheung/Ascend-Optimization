# `mse_loss_backward`：Ascend 910B 阶段报告

> 历史归档：Excel 华为列 G7=1.1688×，已超过 0.8×，退出当前优化主线。下文记录当时实验，当前状态见 [归档入口](../../README.md)。

## KG 无 profiler

正式运行通过 42/42 workload，几何平均加速比 1.326x，最差 1.064x。候选为融合的 Triton elementwise kernel，按输入规模选择 block，并保留非连续输入 fallback。

## profiler 版本

正式运行：`mse-loss-backward-profile-20261007`。42 项合计通过，包括 27 项正确性与 15 项计时；几何平均加速比 1.243x，最差 0.937x。它总体快于 PyTorch，但低于无 profiler 的 1.326x。

## 复现证据

- 无 profiler：`runtime/kg-controller/runs/mse-loss-backward-noprofile-63c4721c`
- profiler：`runtime/kg-controller/runs/mse-loss-backward-profile-20261007`
- profiler 正式结果：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`
- profiler 运行请求启用了 `--profile`，设备为 Ascend 910B。

## 评测口径更新（2026-10-09）

上述加速比均以 Ascend PyTorch 为基线，候选来自 KG 运行。启用 `--profile` 和产生性能提升是两个不同结论。与 FlagGems 原实现的同合同实测及源码差异见 [原版对照报告](../../../../comparisons/flaggems-comparison-20261009/README.md)。
