---
schema_version: '1.0'
id: kg:method:triton-ascend-uint8-host-conversion
kind: method
title: Convert uint8 inputs before Triton Ascend load and compute
summary: Convert uint8 data to a supported signed integer type on the host before
  launching a Triton Ascend kernel rather than loading uint8 directly.
claim_key: method.triton_ascend.uint8_host_conversion
domains:
- datatype
- implementation
- portability
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:23.810116Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/01-Programming-Model/04-data-types.md#7.2
    避免不支持的类型
  title: Convert uint8 inputs before Triton Ascend load and compute
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: triton
      compiler: triton-ascend
  operator:
    dtypes:
    - uint8
    - int8
  numerics:
    exact: false
retrieval:
  phases:
  - initial
  - post_error
  tasks:
  - constraint_check
  - implementation
  - diagnosis
  - portability_analysis
  symptoms:
  - unsupported_uint8
  - load_dtype_error
  - compile_error
  techniques:
  - host_dtype_conversion
  - signed_integer_substitution
  keywords:
  - uint8
  - int8
  - host conversion
  - tl.load
  - unsupported dtype
evidence_state: source_supported
managed:
  content_hash: sha256:2ae67edcf06135a01590fbaa0ff42213c59606f6f519e97933a7edb6455455cc
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:23.810116Z'
  updated_at: '2026-07-27T09:21:23.810116Z'
---

# Claim

Convert uint8 data to a supported signed integer type on the host before launching a Triton Ascend kernel rather than loading uint8 directly.

# Source material

#### 7.2 避免不支持的类型

```python
# 错误: 使用 uint8 类型
# x = tl.load(x_ptr + offsets, mask=mask)  # x_ptr 指向 uint8 数据 -> 不支持!

# 正确: 使用 int8 代替 uint8
x_int8 = tl.load(x_ptr_int8 + offsets, mask=mask)

# 如果确实需要处理 uint8 数据，先在 host 端转换
# x_npu = x_original.to(torch.int8).npu()
```

# Applicability

The structured scope on this Concept is normative.
