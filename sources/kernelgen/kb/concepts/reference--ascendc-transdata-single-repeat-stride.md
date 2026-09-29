---
schema_version: '1.0'
id: kg:reference:ascendc-transdata-single-repeat-stride
kind: reference
title: Set TransDataTo5HD repeat strides to zero for a single repeat
summary: Small-tile TransDataTo5HD calls with repeats equal to one must set source
  and destination repeat strides to zero.
claim_key: reference.ascendc.transdata_single_repeat_stride
domains:
- constraint
- language
- layout
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-transpose.md#2.2 `repeats
    == 1` 时 stride 必须置 0
  title: Set TransDataTo5HD repeat strides to zero for a single repeat
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: ascendc
  operator:
    motifs:
    - transpose
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
  - small_tile_failure
  - transpose_wrong_result
  techniques:
  - stride_configuration
  keywords:
  - TransDataTo5HD
  - repeats
  - stride
  - dstRS
  - srcRS
evidence_state: source_supported
managed:
  content_hash: sha256:ca0fd160740a282b5eae38986bcec1ce56293b618687401de85cd3c6a94e0965
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Small-tile TransDataTo5HD calls with repeats equal to one must set source and destination repeat strides to zero.

# Source material

### 2.2 `repeats == 1` 时 stride 必须置 0

```cpp
uint16_t dstRS = (repeats == 1) ? 0 : 16;
uint16_t srcRS = (repeats == 1) ? 0 : 1;
```

这是小 tile 场景的硬约束，不能省。

# Applicability

The structured scope on this Concept is normative.
