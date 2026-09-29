---
schema_version: '1.0'
id: kg:method:ascendc-datacopypad-selection
kind: method
title: Prefer DataCopyPad for uncertain or non-aligned AscendC transfers
summary: Use DataCopyPad for transfers whose 32-byte alignment is not guaranteed,
  reserve DataCopy for strictly aligned sizes, and use bulk transfers rather than
  scalar GlobalTensor access.
claim_key: method.ascendc.datacopypad_selection
domains:
- implementation
- memory
- optimization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-datacopy.md#选择规则
  title: Prefer DataCopyPad for uncertain or non-aligned AscendC transfers
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
  - post_profile
  tasks:
  - constraint_check
  - implementation
  - diagnosis
  - next_experiment
  symptoms:
  - unaligned_transfer
  - scalar_memory_access
  - boundary_error
  techniques:
  - DataCopyPad
  - bulk_transfer
  keywords:
  - DataCopy
  - DataCopyPad
  - SetValue
  - GetValue
  - alignment
evidence_state: source_supported
managed:
  content_hash: sha256:beaee3f4e3dc0d898d572a9389d7225eb907df8665d1c8d8f9990396c6809d90
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Use DataCopyPad for transfers whose 32-byte alignment is not guaranteed, reserve DataCopy for strictly aligned sizes, and use bulk transfers rather than scalar GlobalTensor access.

# Source material

## 选择规则

**原则：优先使用 DataCopyPad**

| 场景 | API | 原因 |
|-----|-----|------|
| **非对齐或不确定对齐** | `DataCopyPad` | 自动处理对齐/非对齐，避免边界 bug |
| **数据量严格 32 字节对齐** | `DataCopy` 或 `DataCopyPad` | 确定对齐时 DataCopy 可用，DataCopyPad 更安全 |

### ⛔️ 黑名单 API（禁止在生产代码中使用）

| API | 禁止原因 | 仅允许场景 |
|-----|---------|-----------|
| `GlobalTensor::SetValue(idx, val)` | 效率极低，单元素逐个写入 | **仅调试时使用** |
| `GlobalTensor::GetValue(idx)` | 效率极低，单元素逐个读取 | **仅调试时使用** |

```cpp
// ❌ 禁止：生产代码使用 SetValue/GetValue
for (uint32_t i = 0; i < size; i++) {
    xGm.SetValue(i, value);    // ⛔️ 效率极低
    T val = xGm.GetValue(i);   // ⛔️ 效率极低
}

// ✅ 正确：使用 DataCopyPad 批量搬运
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

// ✅ 允许：调试时单点验证
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));  // 仅调试
```

**为什么优先 DataCopyPad？**

1. 自动处理非对齐，无需手动判断
2. CopyIn 和 CopyOut 都适用
3. Tiling 设计时可能产生非对齐的 tile 大小
4. 对齐场景下性能差异可忽略

### DataCopyPad 参数选择：统一使用 Ext 版本

`DataCopyPad` 同时接受 `DataCopyParams` 和 `DataCopyExtParams`，两者底层走**同一条硬件指令**，性能无差异。统一使用 Ext 版本：

| 参数 | 非 Ext 版本 | Ext 版本 | 推荐 |
|------|-----------|---------|------|
| 搬运参数 | `DataCopyParams`（uint16_t，blockLen 最大 65535） | `DataCopyExtParams`（uint32_t，blockLen 最大 2097151） | **Ext** |
| 填充参数 | `DataCopyPadParams`（paddingValue 为 uint64_t） | `DataCopyPadExtParams<T>`（paddingValue 为 T 类型） | **Ext** |

**理由**：
1. 参数范围更大，避免溢出风险
2. `DataCopyPadExtParams<T>` 填充值类型安全，编译期检查
3. 性能完全一致（底层同一条 MTE 指令）
4. 唯一代价是多写一个 `rsv=0` 字段

```cpp
// ❌ 不推荐：DataCopyParams 参数范围有限，paddingValue 类型不安全
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPadParams padParams{true, 0, padElements, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);

// ✅ 推荐：统一使用 Ext 版本
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPadExtParams<float> padParams{true, 0, padElements, 0.0f};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

---

# Applicability

The structured scope on this Concept is normative.
