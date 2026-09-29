---
schema_version: '1.0'
id: kg:diagnostic:ascendc-reduce-row-offset-alignment
kind: diagnostic
title: Diagnose multi-row AscendC Reduce failures from an unaligned row stride
summary: When a row reduction works for one row but fails for multiple rows, compute
  each source offset with the aligned row length rather than the valid element count.
claim_key: diagnostic.ascendc.reduce_row_offset_alignment
domains:
- correctness
- diagnosis
- reduction
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-reduce.md#错误2：rowOffset 用
    rLength 而非 rLengthAlign
  title: Diagnose multi-row AscendC Reduce failures from an unaligned row stride
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: ascendc
  operator:
    motifs:
    - reduction
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
  - single_row_pass_multi_row_fail
  - wrong_result
  - row_stride_error
  techniques:
  - aligned_row_stride
  keywords:
  - rowOffset
  - rLength
  - rLengthAlign
  - Reduce
  - multi-row
evidence_state: source_supported
managed:
  content_hash: sha256:6f0f488d9515ba090b8f1feffa9ad3fc5bbec67f2bc127802bff4cc7606eac43
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

When a row reduction works for one row but fails for multiple rows, compute each source offset with the aligned row length rather than the valid element count.

# Source material

### 错误2：rowOffset 用 rLength 而非 rLengthAlign

```cpp
// ❌ 错误：单行通过，多行失败
uint32_t rowOffset = rowIdx * rLength;

// ✅ 正确
uint32_t rowOffset = rowIdx * rLengthAlign;
```

# Applicability

The structured scope on this Concept is normative.
