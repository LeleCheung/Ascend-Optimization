---
schema_version: '1.0'
id: kg:reference:ascendc-vector-repeat-time-range
kind: reference
title: Bound AscendC Vector API repeatTime by its parameter type
summary: A Vector API repeatTime declared as uint8_t accepts at most 255 iterations,
  and larger values wrap or truncate and can silently skip computation.
claim_key: reference.ascendc.vector_repeat_time_range
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
  locator: ops/ascendc-api-best-practices/references/api-repeat-limits.md#核心约束
  title: Bound AscendC Vector API repeatTime by its parameter type
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
  - repeat_overflow
  - silent_wrong_result
  - boundary_256_failure
  techniques:
  - parameter_range_check
  keywords:
  - repeatTime
  - uint8
  - '255'
  - overflow
  - Vector API
evidence_state: source_supported
managed:
  content_hash: sha256:95db10be766d4d1d94ae7684d8619656323667911984e2c22196bfb7039ce21d
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

A Vector API repeatTime declared as uint8_t accepts at most 255 iterations, and larger values wrap or truncate and can silently skip computation.

# Source material

## 核心约束

**使用 Vector API 前，必须确认 `repeatTime` 参数的范围限制！**

当 `repeatTime` 为 `uint8_t` 类型时，最大值 **255**。

---

# Applicability

The structured scope on this Concept is normative.
