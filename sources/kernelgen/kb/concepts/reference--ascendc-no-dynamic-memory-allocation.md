---
schema_version: '1.0'
id: kg:reference:ascendc-no-dynamic-memory-allocation
kind: reference
title: AscendC AI Core kernels cannot dynamically allocate memory
summary: AI Core kernels must use statically planned buffers and cannot use std::vector,
  new, or malloc for runtime allocation.
claim_key: reference.ascendc.no_dynamic_memory_allocation
domains:
- constraint
- language
- memory
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-restrictions.md#1.2 禁止动态内存分配
  title: AscendC AI Core kernels cannot dynamically allocate memory
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
  - architecture_selection
  - implementation
  - diagnosis
  symptoms:
  - compile_error
  - dynamic_allocation
  techniques:
  - static_buffer_planning
  keywords:
  - dynamic memory
  - std vector
  - new
  - malloc
  - InitBuffer
evidence_state: source_supported
managed:
  content_hash: sha256:dc2eece82c1673b8bb8893a61088f7a553226ac2a00e386ab6b23afdb28c6a37
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

AI Core kernels must use statically planned buffers and cannot use std::vector, new, or malloc for runtime allocation.

# Source material

### 1.2 禁止动态内存分配

**原因**：AI Core 无动态内存管理能力

**触发场景**：创建数组、缓冲区等

**错误示例**：
```cpp
std::vector<int> vec;       // ❌ 动态分配
int* ptr = new int[10];     // ❌ 动态分配
int* arr = malloc(100);     // ❌ 动态分配
```

**正确替代**：使用静态分配
```cpp
int arr[10];                          // ✅ 栈分配（Host 侧）
constexpr uint32_t SIZE = 1024;       // ✅ 编译期常量
pipe.InitBuffer(inQueue, 2, SIZE);    // ✅ UB 静态分配（Kernel 侧）
```

# Applicability

The structured scope on this Concept is normative.
