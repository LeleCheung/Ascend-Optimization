---
schema_version: '1.0'
id: kg:reference:triton-ascend-int8-dot-contract
kind: reference
title: Use int32 accumulation and output for Triton Ascend int8 dot
summary: Triton Ascend tl.dot accepts int8 matrix inputs and uses int32 accumulation/output
  rather than low-precision integer accumulation.
claim_key: reference.triton_ascend.int8_dot_contract
domains:
- language
- matrix-multiply
- numerics
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:14.771517Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/zh/triton_api/Linear_Algebra_Ops/dot.md#2.1 参数说明
  title: Use int32 accumulation and output for Triton Ascend int8 dot
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
  - initial
  - post_error
  - post_evaluation
  tasks:
  - constraint_check
  - architecture_selection
  - implementation
  - diagnosis
  symptoms:
  - dot_dtype_error
  - accumulator_mismatch
  techniques:
  - int32_accumulation
  keywords:
  - int8
  - tl.dot
  - int32
  - accumulator
  - out_dtype
evidence_state: source_supported
managed:
  content_hash: sha256:afff2da87a9e5ef89260f2f4d4bb6f55ebe01709e15c64c925f0c36013dc6bd3
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:14.771517Z'
  updated_at: '2026-07-27T09:21:14.771517Z'
---

# Claim

Triton Ascend tl.dot accepts int8 matrix inputs and uses int32 accumulation/output rather than low-precision integer accumulation.

# Source material

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `input`        | `int8 fp16 bf16 fp32`     |     第一个输入，2D or 3D 张量， 为了避免溢出 取值范围限制为-5-5     |
| `other`       | `int8 fp16 bf16 fp32`     |     第二个输入,  2D or 3D 张量，为了避免溢出 取值范围限制为-5-5    |
| `acc`           | `int32  float32`    | 存累加结果的张量, accumulator tensor. If not None, the result is added to this tensor, acc_dtype支持 {:code:`float16`, :code:`float32`, :code:`int32`} |
| `input_precision`   | -                 |  Available options for NVIDIA 通过选择精度模式来决定是否启用 Tensor Cores 加速    |
| `max_num_imprecise_acc`     | `int`    | 多少次低精度的累加数（当前昇腾不支持低精度累加） |
| `out_dtype`     | `fp32  int32`    | 输出结果类型|

返回值：
`tl.tensor`：矩阵乘结果

# Applicability

The structured scope on this Concept is normative.
