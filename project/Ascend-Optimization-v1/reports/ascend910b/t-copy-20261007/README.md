# `t_copy`：Ascend 910B 阶段报告

## KG 无 profiler

正式运行通过 15/15 workload，几何平均加速比 2.007x。64×64 小尺寸约 5.7～6.3x；4096×4096 大尺寸约 0.65～0.72x，说明大尺寸仍受 tile 和转置布局限制。

## profiler 版本

`t-copy-profile-20261007` 启用 `--profile` 后通过 15/15 正确性和 15/15 计时，几何平均加速比 3.481x，最差 1.526x。结果文件：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`。

## 结论

候选已正确且总体快于 PyTorch，但大尺寸仍需更大的 tile 或每个 program 处理多个列块；当前 profiler 结果没有改变这一判断。
