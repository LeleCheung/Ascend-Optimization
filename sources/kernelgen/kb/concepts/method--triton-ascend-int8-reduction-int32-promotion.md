---
schema_version: '1.0'
id: kg:method:triton-ascend-int8-reduction-int32-promotion
kind: method
title: Promote Triton Ascend int8 reductions to int32
summary: Convert int8 and int16 values to tl.int32 before sum or other overflow-sensitive
  reductions instead of reducing directly in the narrow integer type.
claim_key: method.triton_ascend.int8_reduction_int32_promotion
domains:
- implementation
- numerics
- reduction
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:23.810116Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/triton-latency-optimizer/references/docs_triton_IR/docs_for_triton_agent/08-data-type-precision.md#2.1
    核心规则：所有归约必须在 FP32 下进行
  title: Promote Triton Ascend int8 reductions to int32
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: triton
      compiler: triton-ascend
  operator:
    motifs:
    - reduction
    dtypes:
    - int8
    - int16
  numerics:
    exact: false
retrieval:
  phases:
  - initial
  - post_error
  - post_evaluation
  tasks:
  - constraint_check
  - architecture_selection
  - implementation
  - diagnosis
  symptoms:
  - integer_overflow
  - reduction_wrong_result
  techniques:
  - int32_promotion
  - wide_accumulation
  keywords:
  - int8
  - int16
  - int32
  - reduction
  - tl.sum
  - overflow
evidence_state: source_supported
managed:
  content_hash: sha256:e96198f14cb099cf8fe499ce6e17f8e9d02ceb47cbdeb4512f57756ed0852728
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:23.810116Z'
  updated_at: '2026-07-27T09:21:23.810116Z'
---

# Claim

Convert int8 and int16 values to tl.int32 before sum or other overflow-sensitive reductions instead of reducing directly in the narrow integer type.

# Source material

### 2.1 核心规则：所有归约必须在 FP32 下进行

NPU 上归约操作（`tl.sum`、`tl.max`、`tl.min`、`tl.argmax`、`tl.argmin`）的精度行为与 GPU 不同，必须手动确保在 FP32 精度下执行归约。

| 输入类型 | GPU 行为 | NPU (910_95) 行为 | 正确做法 |
|---------|---------|-------------------|---------|
| fp16 | 自动提升为 fp32 归约 | `tl.sum` 直接 fp16 归约；`tl.max`/`tl.min` 自动提升为 fp32 | `tl.sum` 需手动 `.to(tl.float32)` 后归约 |
| bf16 | 直接 bf16 归约 | 自动提升为 fp32 归约 | 无需额外处理（编译器自动提升） |
| int8 | 提升为 int32 归约 | 直接 int8 归约 | 手动 `.to(tl.int32)` 后归约 |
| int16 | 提升为 int32 归约 | 直接 int16 归约 | 手动 `.to(tl.int32)` 后归约 |

# Applicability

The structured scope on this Concept is normative.
