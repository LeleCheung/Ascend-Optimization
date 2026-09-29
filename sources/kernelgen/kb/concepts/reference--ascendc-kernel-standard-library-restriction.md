---
schema_version: '1.0'
id: kg:reference:ascendc-kernel-standard-library-restriction
kind: reference
title: AscendC kernel code cannot use C++ standard-library compute functions
summary: AI Core kernel code must replace std math and comparison functions with AscendC
  vector APIs or supported scalar expressions.
claim_key: reference.ascendc.kernel_standard_library_restriction
domains:
- compiler
- constraint
- language
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: 'ops/ascendc-api-best-practices/references/api-restrictions.md#1.1 禁止使用
    std:: 计算函数'
  title: AscendC kernel code cannot use C++ standard-library compute functions
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: ascendc
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
  symptoms:
  - compile_error
  - unsupported_standard_library
  techniques:
  - api_substitution
  keywords:
  - std
  - AscendC
  - kernel
  - Abs
  - Min
  - Max
  - Sqrt
  - Exp
  - Log
evidence_state: source_supported
managed:
  content_hash: sha256:b51fe5c44d237ef30da3a5d227d04c3c50b8f8d2f37127b206ef9822ba591905
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

AI Core kernel code must replace std math and comparison functions with AscendC vector APIs or supported scalar expressions.

# Source material

### 1.1 禁止使用 std:: 计算函数

**原因**：Kernel 侧不支持 C++ 标准库，必须使用 Ascend C 提供的专用 API

**触发场景**：所有数学计算、比较操作

**禁止列表**：

| std:: 函数 | ❌ 错误用法 | ✅ Ascend C 替代 | 说明 |
|-----------|----------|----------------|------|
| `std::abs` | `std::abs(x)` | `AscendC::Abs(dst, src, count)` | 绝对值 |
| `std::min/max` | `std::min(a, b)` | `(a < b) ? a : b` 或 `AscendC::Min/Max` | 最小/最大值 |
| `std::sqrt` | `std::sqrt(x)` | `AscendC::Sqrt(dst, src, count)` | 平方根 |
| `std::pow` | `std::pow(x, y)` | `AscendC::Power(dst, src, count)` | 幂运算 |
| `std::exp` | `std::exp(x)` | `AscendC::Exp(dst, src, count)` | 指数 |
| `std::log/log2/log10` | `std::log(x)` | `AscendC::Log/Log2/Log10(dst, src, count)` | 对数 |
| `std::sin/cos/tan` | `std::sin(x)` | `AscendC::Sin/Cos/Tan(dst, src, count)` | 三角函数 |
| `std::floor/ceil/round` | `std::floor(x)` | `AscendC::Floor/Ceil/Round(dst, src, count)` | 取整 |
| `std::isnan/isinf` | `std::isnan(x)` | 手动检查 | 特殊值判断 |

**错误示例**：
```cpp
#include <algorithm>
#include <cmath>

uint32_t result = std::min(a, b);  // ❌ 编译错误
float val = std::sqrt(x);          // ❌ 编译错误
float val = std::exp(x);           // ❌ 编译错误
```

**正确替代**：
```cpp
// min/max：使用三元操作符
uint32_t result = (a < b) ? a : b;  // ✅ min
uint32_t result = (a > b) ? a : b;  // ✅ max

// 或使用 Ascend C API（批量操作）
AscendC::LocalTensor<T> minLocal = minBuf.Get<T>();
AscendC::LocalTensor<T> srcLocal = srcBuf.Get<T>();
AscendC::Min<T>(minLocal, srcLocal, src2Local, count);  // ✅ 批量最小值

// sqrt/exp/log 等：使用 Ascend C API
AscendC::LocalTensor<T> dstLocal = dstBuf.Get<T>();
AscendC::LocalTensor<T> srcLocal = srcBuf.Get<T>();
AscendC::Sqrt<T>(dstLocal, srcLocal, count);  // ✅ 平方根
AscendC::Exp<T>(dstLocal, srcLocal, count);   // ✅ 指数
AscendC::Log<T>(dstLocal, srcLocal, count);   // ✅ 对数
```

**⚠️ 重要**：所有数学计算都必须使用 Ascend C API，不能混用 std:: 函数！

# Applicability

The structured scope on this Concept is normative.
