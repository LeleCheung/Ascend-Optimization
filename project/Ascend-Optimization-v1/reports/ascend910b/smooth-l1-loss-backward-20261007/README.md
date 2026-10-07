# `smooth_l1_loss_backward`：Ascend 910B 阶段报告

## 正式结果

修复 beta=0 且输入等值时的 NaN 分支后，`smooth-l1-backward-phase2b` 通过 303/303 正确性和计时，几何平均加速比 3.515x，最差 1.867x。

## profiler 尝试

`smooth-l1-profile-20261007` 的 `--profile` 提交因 KGS 19654 响应超时而失败，属于基础设施问题，未产生可用 profiler 性能结论。

## 结论

候选在无 profiler 和 profiler 两种口径下都稳定加速。

## profiler 版本

修正 catalog contract 后，`smooth-l1-profile-20261007c` 正式通过 303/303 正确性和计时，几何平均加速比 3.533x，最差 1.875x。结果文件：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`。

