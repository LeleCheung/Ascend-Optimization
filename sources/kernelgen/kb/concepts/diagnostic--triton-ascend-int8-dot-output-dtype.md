---
schema_version: '1.0'
id: kg:diagnostic:triton-ascend-int8-dot-output-dtype
kind: diagnostic
title: Diagnose unsupported low-precision out_dtype in Triton Ascend dot
summary: When an int8 tl.dot fails around out_dtype or accumulator typing, use int32
  output and accumulation because Ascend does not support int8 or fp16 dot output
  and does not support imprecise accumulation controls.
claim_key: diagnostic.triton_ascend.int8_dot_output_dtype
domains:
- compiler
- diagnosis
- matrix-multiply
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:14.771517Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/zh/triton_api/Linear_Algebra_Ops/dot.md#2.3 特殊限制说明
  title: Diagnose unsupported low-precision out_dtype in Triton Ascend dot
scope:
  target:
    level: architecture
    backend: ascend
    architecture: DAV_2201
    software:
      language: triton
      compiler: triton-ascend
  operator:
    motifs:
    - matrix_multiply
    dtypes:
    - int8
  numerics:
    exact: false
retrieval:
  phases:
  - post_error
  - post_evaluation
  tasks:
  - diagnosis
  - implementation
  - next_experiment
  symptoms:
  - dot_dtype_error
  - unsupported_out_dtype
  - compile_error
  techniques:
  - int32_accumulation
  - output_cast
  keywords:
  - int8
  - tl.dot
  - out_dtype
  - int32
  - max_num_imprecise_acc
evidence_state: source_supported
managed:
  content_hash: sha256:903418870d0bf88386bae19eccd95252633b5267516dcc32d8c0ad8d0d36e955
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:14.771517Z'
  updated_at: '2026-07-27T09:21:14.771517Z'
---

# Claim

When an int8 tl.dot fails around out_dtype or accumulator typing, use int32 output and accumulation because Ascend does not support int8 or fp16 dot output and does not support imprecise accumulation controls.

# Source material

### 2.3 特殊限制说明

- Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

- acc 不能支持fp16，为了精度硬件默认就是fp32

- max_num_imprecise_acc 暂时不支持

- out_dtype对比GPU 缺乏int8和FP16的类型支持

# Applicability

The structured scope on this Concept is normative.
