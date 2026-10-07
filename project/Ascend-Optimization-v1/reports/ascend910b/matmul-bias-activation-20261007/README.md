# matmul_bias_activation 阶段报告

## 结果

本次 KG 无 profiler 任务生成了融合矩阵乘、bias 和 ReLU 的 Triton 候选，但正确性阶段在候选覆盖检查处终止，未进入性能测量，因此没有可报告的加速比。

## 证据

- 910B workspace：`runtime/kg-controller/runs/matmul-bias-phase1-skip/`
- 状态：`FAILED / optimize`
- eval 状态：`RUNTIME_ERROR`
- 正确性 timing workload：`0`
- 性能：未测量

## 根因

冻结的 correctness 测试通过 `flag_gems.fused.matmul_bias_activation` 子模块直接导入实现，而 KG 的 override 只注入顶层 `flag_gems.matmul_bias_activation`。因此 27 个正确性 case 都没有调用候选，覆盖检查拒绝继续。需要扩展 adapter 的子模块注入，或提供调用顶层导出的 correctness 测试后再重跑。
