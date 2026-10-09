# `_prelu_kernel_backward`：Ascend 910B 阶段报告

> 历史归档：Excel 华为列 G33=1.8152×，已超过 0.8×，退出当前优化主线。下文记录当时实验，当前状态见 [归档入口](../../README.md)。

## 结果

- 正式 KG 运行：`prelu-underscore-phase7`
- 30 项合计通过：正确性 21/21，计时 9/9；代码审查通过。
- 几何平均加速比：**2.797x**；最差：**1.984x**。
- fp16：约 3.03x、3.27x、3.70x；fp32：约 1.98x、3.02x、3.22x；bf16：约 2.40x～2.50x。

候选实现将 `grad_input` 与梯度权重的分块累加融合，梯度权重使用 fp32 的确定性分块归约；标量权重和按通道权重分别处理。设备实测覆盖空输入、2D/4D/5D 形状及三种浮点类型。

## 复现证据

- 910B 工作区：`/data/hanle/ascend-optimization/runtime/kg-controller/runs/prelu-underscore-phase7`
- 正式结果：`stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`
- 结果状态：`PASSED`，`num_workloads=30`，`num_passed=30`，`geo_mean=2.7974668629`，`min_speedup=1.9841713328`。

## profiler 版本

`prelu-underscore-profile-20261007b` 启用 `--profile` 后正式通过 30 项合计评测，几何平均加速比 2.835x，最差 1.960x。结果文件位于该工作区的 `stages/optimize/work/1R/agent0/.kernelgen/evals/round-0001/result.json`。

## 测试修正

原测试的 per-channel weight 使用了 `shape[-1]`，而 Ascend 原生算子要求权重长度为 1 或 `shape[1]`。这会使 reference 在候选尚未调用前直接报错，造成零 workload 的基础设施失败。已改为 `shape[1]`，随后重新生成 catalog 快照并完成正式评测。

## 评测口径更新（2026-10-09）

上述数值是候选相对 Ascend PyTorch 的加速比。候选还按本机 NPU 接口将权重梯度归约到 `weight.shape`，与 FlagGems 通用版本的逐元素输出不同，不能直接当作纯性能改进。与原版的同合同实测见 [原版对照报告](../../../../comparisons/flaggems-comparison-20261009/README.md)。
