---
schema_version: '1.0'
id: kg:reference:triton-ascend-int8-cast-overflow-modes
kind: reference
title: Triton Ascend integer Cast supports truncation and saturation modes
summary: Triton Ascend extends integer Cast with trunc as the default overflow behavior
  and saturate for clamping values to the destination integer range.
claim_key: reference.triton_ascend.int8_cast_overflow_modes
domains:
- language
- numerics
- quantization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:14.771517Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/zh/triton_api/Creation_Ops/cast.md#1 功能作用说明
  title: Triton Ascend integer Cast supports truncation and saturation modes
scope:
  target:
    level: architecture
    backend: ascend
    architecture: DAV_2201
    software:
      language: triton
      compiler: triton-ascend
  operator:
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
  - implementation
  - diagnosis
  symptoms:
  - integer_overflow
  - quantization_mismatch
  techniques:
  - saturating_cast
  - overflow_mode
  keywords:
  - int8
  - cast
  - overflow_mode
  - trunc
  - saturate
evidence_state: source_supported
managed:
  content_hash: sha256:f78a4bd024ae86b5052e43c1def27861143af9f56d19e72caf057a9886298058
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:14.771517Z'
  updated_at: '2026-07-27T09:21:14.771517Z'
---

# Claim

Triton Ascend extends integer Cast with trunc as the default overflow behavior and saturate for clamping values to the destination integer range.

# Source material

## 1 功能作用说明

将张量转换为指定的数据类型，支持数值类型转换、位级别重解释（bitcast）、浮点降精度舍入模式，以及Ascend扩展的整数溢出处理模式。

**语法：**

- `triton.language.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)` - 函数调用形式
- `input.cast(dtype, fp_downcast_rounding=None, bitcast=False)` - 成员函数形式

**功能：**

- 数值类型转换：整型<->整型、浮点<->浮点、整型<->浮点
- 位级别重解释（bitcast）：不改变比特，只改变解释类型
- 浮点降精度支持舍入模式：`rtne`（默认，四舍六入五成双）、`rtz`（向零）
- 整数转换（Ascend 扩展）支持溢出模式：`trunc`（截断，默认）、`saturate`（饱和）

# Applicability

The structured scope on this Concept is normative.
