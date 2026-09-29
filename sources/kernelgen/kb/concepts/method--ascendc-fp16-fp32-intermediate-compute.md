---
schema_version: '1.0'
id: kg:method:ascendc-fp16-fp32-intermediate-compute
kind: method
title: Use FP32 intermediates for numerically sensitive AscendC reductions
summary: Widen FP16 or BF16 inputs to FP32 for sensitive reductions and nonlinear
  operations, then round back to the output type after computation.
claim_key: method.ascendc.fp16_fp32_intermediate_compute
domains:
- implementation
- numerics
- reduction
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-precision.md#混合精度计算模式（FP16
    输入）
  title: Use FP32 intermediates for numerically sensitive AscendC reductions
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: ascendc
  operator:
    motifs:
    - reduction
    - normalization
    dtypes:
    - float16
    - bfloat16
  numerics:
    exact: false
retrieval:
  phases:
  - initial
  - post_error
  - post_evaluation
  tasks:
  - architecture_selection
  - implementation
  - diagnosis
  - next_experiment
  symptoms:
  - precision_failure
  - overflow
  - accumulation_error
  techniques:
  - fp32_accumulation
  - mixed_precision
  keywords:
  - FP16
  - BF16
  - FP32
  - Softmax
  - LayerNorm
  - accumulation
evidence_state: source_supported
managed:
  content_hash: sha256:89779557a137dca4940bd3c91413572db03b591260548d172271d291f7851e27
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Widen FP16 or BF16 inputs to FP32 for sensitive reductions and nonlinear operations, then round back to the output type after computation.

# Source material

## 混合精度计算模式（FP16 输入）

### 适用场景

当输入输出为 FP16，但需要 FP32 精度进行中间计算时（如 Softmax、LayerNorm）。

### 计算流程

```
half 输入 → Cast(FP32) → 中间计算(FP32) → Cast(half) → half 输出
```

### 为什么需要 FP32 中间计算？

1. **ReduceMax/Exp/ReduceSum** 在 FP32 上精度更稳定
2. **避免 FP16 数值溢出**：Exp 结果可能超出 FP16 表示范围
3. **累积误差控制**：多次运算的累积误差在 FP32 下更小

### 加减法场景示例

半精度加减法默认升 FP32；仅当 spec 明确"输入同量级"（如 mask 叠加、已归一化概率相加）时才允许直接 `Add/Sub<half>`。BF16 与 FP16 适用同一规则，仅临界比值不同（BF16=128，FP16=1024）。

> 完整示例、决策表与 Kernel 集成要点见 [api-arithmetic.md → 场景3](api-arithmetic.md#场景3半精度加减法精度优化)。

---

# Applicability

The structured scope on this Concept is normative.
