# `batch_norm_backward`：Ascend 910B 阶段报告

## 结果

- 正式 KG 运行：`batch-norm-adapter-phase6`
- 正确性：45/45；计时：45/45；代码审查通过。
- 几何平均加速比：**4.937x**；最差：**2.149x**。
- fp16、fp32、bf16 均通过；训练和推理分支、三种输出组合均覆盖。
- 单项加速比约 2.15x～7.49x，小矩阵主要收益来自减少 kernel launch，大平面使用分块归约。

## 复现证据

- 910B 工作区：`/data/hanle/ascend-optimization/runtime/kg-controller/runs/batch-norm-adapter-phase6`
- 正式结果：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`
- 结果状态：`PASSED`，`num_workloads=45`，`num_passed=45`，`geo_mean=4.9373094682`，`min_speedup=2.1493979821`。

## 测试修正

原测试用 `torch.randn(C)` 生成 saved variance，负值会使 reference 和候选同时产生 NaN，而断言使用 `equal_nan=False`，导致无意义的 correctness 失败。已改为正方差 `torch.rand(C) + 0.5`，保持 batch norm 的有效输入域后重新生成快照并完成正式评测。
