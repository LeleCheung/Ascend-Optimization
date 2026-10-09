# `batch_norm_backward`：Ascend 910B 阶段报告

> 历史归档：Excel 华为列 G159=4.3885×，已超过 0.8×，退出当前优化主线。下文记录当时实验，当前状态见 [归档入口](../../README.md)。

## 结果

- 正式 KG 运行：`batch-norm-adapter-phase6`
- 正确性 30/30、计时 15/15，合计 45 项；代码审查通过。
- 几何平均加速比：**4.937x**；最差：**2.149x**。
- fp16、fp32、bf16 均通过；训练和推理分支、三种输出组合均覆盖。
- 单项加速比约 2.15x～7.49x，小矩阵主要收益来自减少 kernel launch，大平面使用分块归约。

## 复现证据

- 910B 工作区：`/data/hanle/ascend-optimization/runtime/kg-controller/runs/batch-norm-adapter-phase6`
- 正式结果：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`
- 结果状态：`PASSED`，`num_workloads=45`，`num_passed=45`，`geo_mean=4.9373094682`，`min_speedup=2.1493979821`。

## profiler 尝试

`batch-norm-profile-20261007` 已提交 `--profile` 流程，但 19654 在提交阶段响应超时，状态为基础设施失败，未产生可用性能结论；正式无 profiler 结果不受影响。

## 测试修正

原测试用 `torch.randn(C)` 生成 saved variance，负值会使 reference 和候选同时产生 NaN，而断言使用 `equal_nan=False`，导致无意义的 correctness 失败。已改为正方差 `torch.rand(C) + 0.5`，保持 batch norm 的有效输入域后重新生成快照并完成正式评测。

## 评测口径更新（2026-10-09）

4.937x 是相对 Ascend PyTorch 的历史加速比。候选将本机接口名为 `save_invstd` 的参数按方差处理，计算 `1/sqrt(save_invstd+eps)`；FlagGems 通用实现直接将其作为标准差倒数。这包含平台语义适配，不能全归为性能优化。与原版的同合同实测见 [原版对照报告](../../../../comparisons/flaggems-comparison-20261009/README.md)。
