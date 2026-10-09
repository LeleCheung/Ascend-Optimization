# `t_copy`：Ascend 910B 阶段报告

> 历史归档：Excel 华为列 G43=3.0894×，已超过 0.8×，退出当前优化主线。下文记录当时实验，当前状态见 [归档入口](../../README.md)。

## KG 无 profiler

正式运行通过 15/15 workload，几何平均加速比 2.007x。64×64 小尺寸约 5.7～6.3x；4096×4096 大尺寸约 0.65～0.72x，说明大尺寸仍受 tile 和转置布局限制。

## profiler 版本

`t-copy-profile-20261007` 启用 `--profile` 后通过正确性 9/9、计时 6/6，合计 15 项；几何平均加速比 3.481x，最差 1.526x。结果文件：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`。

## 结论

后续候选已采用较大的 tile、每个 program 处理多个列块和编译内核启动器缓存，历史计时各项均快于 PyTorch。与 FlagGems 原实现的同合同实测见 [原版对照报告](../../../../comparisons/flaggems-comparison-20261009/README.md)；是否超过原版与是否来自自研 profiling，需要分别核验。
