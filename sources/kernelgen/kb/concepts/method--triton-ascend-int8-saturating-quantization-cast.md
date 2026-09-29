---
schema_version: '1.0'
id: kg:method:triton-ascend-int8-saturating-quantization-cast
kind: method
title: Quantize to int8 with an explicit saturating Triton Ascend Cast
summary: Apply scale and zero point in a wider type, then cast to tl.int8 with overflow_mode=saturate
  so out-of-range values clamp to the int8 domain.
claim_key: method.triton_ascend.int8_saturating_quantization_cast
domains:
- implementation
- numerics
- quantization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:14.771517Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/zh/triton_api/Creation_Ops/cast.md#2.5 使用方法
  title: Quantize to int8 with an explicit saturating Triton Ascend Cast
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
    - elementwise
    - quantization
    dtypes:
    - float32
    - int8
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
  - integer_overflow
  - quantization_mismatch
  techniques:
  - saturating_cast
  - affine_quantization
  keywords:
  - int8
  - quantization
  - scale
  - zero_point
  - saturate
  - tl.cast
evidence_state: source_supported
managed:
  content_hash: sha256:e7cdf35f746a094d091be4f8c59ea2236f7bd5e94561204379249d81dea21412
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:14.771517Z'
  updated_at: '2026-07-27T09:21:14.771517Z'
---

# Claim

Apply scale and zero point in a wider type, then cast to tl.int8 with overflow_mode=saturate so out-of-range values clamp to the int8 domain.

# Source material

### 2.5 使用方法

**基本用法：**

```python
import triton
import triton.language as tl

@triton.jit
def cast_example():
    # 创建float32张量
    x = tl.zeros([2, 3], dtype=tl.float32)

    # 转换为int32
    y = tl.cast(x, tl.int32)

    return y

## 调用示例
result = cast_example()
print(result.dtype)  # 输出: int32
```

**高级用法：**

```python
@triton.jit
def cast_advanced_example():
    # 创建float32张量
    x = tl.zeros([2, 3], dtype=tl.float32)

    # 位级别重解释
    y = x.cast(tl.int32, bitcast=True)

    # 浮点降精度，向零舍入
    z = x.cast(tl.float16, fp_downcast_rounding="rtz")

    # float32 → int8，启用饱和模式（Ascend 扩展，超出 int8 范围的值会被截断到 [-128, 127]）
    w = x.cast(tl.int8, overflow_mode="saturate")

    return y, z, w
```

**实际应用场景：**

```python
@triton.jit
def quantization_kernel(x_ptr, output_ptr, scale, zero_point, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 加载float32数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 量化：转换为int8
    x_quantized = tl.cast(x * scale + zero_point, tl.int8, overflow_mode="saturate")

    # 存储量化结果
    tl.store(output_ptr + offsets, x_quantized, mask=mask)
```

# Applicability

The structured scope on this Concept is normative.
