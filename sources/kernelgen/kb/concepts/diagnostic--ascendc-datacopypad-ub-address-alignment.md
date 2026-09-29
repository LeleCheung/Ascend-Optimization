---
schema_version: '1.0'
id: kg:diagnostic:ascendc-datacopypad-ub-address-alignment
kind: diagnostic
title: Diagnose DataCopyPad failures caused by an unaligned UB start address
summary: When DataCopyPad reports AIV alignment errors or chained AIC timeout errors,
  verify that every per-row UB start offset is 32-byte aligned and stage irregular
  rows into aligned storage.
claim_key: diagnostic.ascendc.datacopypad_ub_address_alignment
domains:
- correctness
- diagnosis
- memory
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-datacopy.md#UB 端起始地址 32B
    对齐（易踩坑）
  title: Diagnose DataCopyPad failures caused by an unaligned UB start address
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
  - post_error
  - post_evaluation
  tasks:
  - diagnosis
  - next_experiment
  symptoms:
  - aiv_error_80
  - ub_address_unaligned
  - aic_timeout
  - data_corruption
  techniques:
  - aligned_staging
  - stride_padding
  keywords:
  - DataCopyPad
  - UB address
  - 32B
  - AIV error 80
  - timeout
evidence_state: source_supported
managed:
  content_hash: sha256:24d2048ca80708a3524c0209c768a0839673f60a687de498fe3b8bea632b422b
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

When DataCopyPad reports AIV alignment errors or chained AIC timeout errors, verify that every per-row UB start offset is 32-byte aligned and stage irregular rows into aligned storage.

# Source material

### UB 端起始地址 32B 对齐（易踩坑）

`DataCopyPad(GM, UB, ...)` 与 `DataCopyPad(UB, GM, ...)` 的 **UB 端起始地址必须 32 字节对齐**（blockLen 可以非 32B 对齐，但起始地址不能）。

按行索引访问 UB buffer 时，行偏移字节数必须是 32 的倍数：

```cpp
// ❌ cols * sizeof(elem) 不是 32 倍数时，row * cols 偏移可能落非对齐地址
// 例如 fp8 + cols=4 时每行 4 字节，只有 row ∈ {0, 8, 16,...} 满足 32B 对齐
DataCopyPad(gmOut[off], ubBuf[row * cols], copyParams);
```

修复方式：引入 strided staging buffer，把不规则行宽数据重排到每行 32B 对齐的连续区：

```cpp
// ✅ 用 strided buf 重排，保证每行 UB src 起址 32B 对齐
auto stridedBuf = strideBuf_.Get<elem_T>();
for (int row = 0; row < mEff; ++row) {
    for (int j = 0; j < cols; ++j) {
        stridedBuf.SetValue(row * 32 + j, ubBuf.GetValue(row * cols + j));
    }
}
DataCopyPad(gmOut[off], stridedBuf[row * 32], copyParams);  // src 每行 32B 对齐
```

**错误码症状**：
- `AIV error 80: The UB address accessed by the VEC instruction is not aligned`
- 连锁触发 `AIC error: timeout or trap error. subErrType: 0x4`

# Applicability

The structured scope on this Concept is normative.
