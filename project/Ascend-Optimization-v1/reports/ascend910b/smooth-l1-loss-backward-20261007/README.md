# `smooth_l1_loss_backward`：Ascend 910B 阶段报告

## 正式结果

修复 beta=0 且输入等值时的 NaN 分支后，`smooth-l1-backward-phase2b` 通过 303/303 正确性和计时，几何平均加速比 3.515x，最差 1.867x。

## profiler 尝试

`smooth-l1-profile-20261007` 的 `--profile` 提交因 KGS 19654 响应超时而失败，属于基础设施问题，未产生可用 profiler 性能结论。

## 结论

候选在无 profiler 的 device-time 口径下稳定加速；需要恢复 profiler 服务后再补采样。
