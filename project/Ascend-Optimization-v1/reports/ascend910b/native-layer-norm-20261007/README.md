# `native_layer_norm`：Ascend 910B 阶段报告

## 导师汇报结论

Triton 候选可以算对，但没有打过 Ascend 原生实现。现有设备测量中，Triton 约为 PyTorch 的 0.18～0.21 倍，原生 CANN/PyTorch 实现约为 0.95～1.05 倍。主要差距来自 Triton kernel 与 host launch 的固定开销，不是 KGS 或设备故障。

## 三个版本

| 版本 | 结果 |
| --- | --- |
| KG 无 profiler | 已完成候选正确性和设备基准；性能明显低于原生实现。 |
| KG 有 profiler | 已生成 baseline、设备探针和 Msprof 数据；正式 profile round 的部分 workload 因返回形状不一致失败。 |
| 基于 profiling 的最终版 | 已完成 profiling 分析和候选修订，但没有得到超过原生实现的正式候选，因此结论是当前 Triton 路径不适合该算子。 |

## 证据

- 无 profiler 工作区：`runtime/kg-controller/runs/native-layer-norm-final-noprofile-v3-20261007`
- profiler 工作区：`runtime/kg-controller/runs/native-layer-norm-final-profile-v3-20261007`
- Msprof 原始数据：`runtime/kg-controller/runs/native-layer-norm-20261007-profile/msprof-pipe/`
- profile round 结果：20 个 workload，10 个通过；失败原因为返回值形状不一致，不能作为性能结论。

后续已修正候选返回的统计量形状（保留 normalized 维的 singleton 轴），但使用 native catalog 重跑时，KG worker 在进入 eval 前因 `AttributeError: 'list' object has no attribute 'get'` 退出；这是 native catalog/worker 的基础设施兼容问题，尚未形成新的正式 profile round。

## 结论

在当前 KernelGen Ascend backend 和 Triton 生成路径下，native_layer_norm 的瓶颈是编译链生成的 kernel/launch 组织方式。若要继续突破，应改进 backend 的融合与 launch 组织，或下沉到 Ascend 更低层 IR/CANN 自定义算子；继续调 Triton block 参数无法弥补当前数量级差距。
