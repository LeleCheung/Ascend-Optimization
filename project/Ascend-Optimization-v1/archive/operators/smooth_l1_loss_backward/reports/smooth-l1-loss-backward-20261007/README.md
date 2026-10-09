# `smooth_l1_loss_backward`：Ascend 910B 阶段报告

> 历史归档：Excel 华为列 G600=2.2139×，已超过 0.8×，退出当前优化主线。下文记录当时实验，当前状态见 [归档入口](../../README.md)。

## 正式结果

修复 beta=0 且输入等值时的 NaN 分支后，`smooth-l1-backward-phase2b` 通过 303 项合计评测，几何平均加速比 3.515x，最差 1.867x。

## profiler 尝试

`smooth-l1-profile-20261007` 的 `--profile` 提交因 KGS 19654 响应超时而失败，属于基础设施问题，未产生可用 profiler 性能结论。

## 结论

候选在两个 KG 运行中均快于 Ascend PyTorch。是否超过 FlagGems 原实现以及 profiling 的实际贡献，需分别对照。

## profiler 版本

修正 catalog contract 后，`smooth-l1-profile-20261007c` 正式通过正确性 291/291、计时 12/12，合计 303 项；几何平均加速比 3.533x，最差 1.875x。结果文件：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`。与原版的同合同实测见 [原版对照报告](../../../../comparisons/flaggems-comparison-20261009/README.md)。

