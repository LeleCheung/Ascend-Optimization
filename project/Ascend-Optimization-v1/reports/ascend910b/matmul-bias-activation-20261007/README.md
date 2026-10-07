# `matmul_bias_activation`：Ascend 910B 阶段报告

## 正式结果

修正 correctness 测试从子模块导入导致候选未注入的问题后，`matmul-bias-phase2` 通过 42/42 正确性和计时，但几何平均加速比仅 0.491x，最差 0.276x，明显慢于 PyTorch。

## profiler 尝试

尚未得到可用 profiler 结果；当前优先级是先解决候选性能低于 PyTorch 的问题。早期 phase1 的失败属于 adapter 注入问题，不能作为性能结论。

## 结论

当前融合 Triton 实现没有达到交付性能目标，应转向 Ascend 原生矩阵乘路径，或重新设计 tile、dtype 和矩阵布局。
