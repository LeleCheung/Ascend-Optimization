---
schema_version: '1.0'
id: kg:method:ascendc-reduce-interface-selection
kind: method
title: Select AscendC Level 2 or Pattern Reduce by reduction structure
summary: Use Level 2 Reduce for independent per-row reductions without alignment requirements
  and Pattern Reduce for aligned cross-row batch reductions.
claim_key: method.ascendc.reduce_interface_selection
domains:
- implementation
- optimization
- reduction
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-reduce.md#接口选择
  title: Select AscendC Level 2 or Pattern Reduce by reduction structure
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
  - initial
  - post_error
  - post_profile
  tasks:
  - architecture_selection
  - implementation
  - diagnosis
  - next_experiment
  symptoms:
  - reduce_api_mismatch
  - alignment_requirement
  techniques:
  - reduce_interface_selection
  keywords:
  - ReduceMax
  - ReduceSum
  - Level 2
  - Pattern
  - AR
  - RA
evidence_state: source_supported
managed:
  content_hash: sha256:1e3c1bc11bf495f03d25c0ef14d7d6a3c0a689580b9623ec8608563edabd2a02
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Use Level 2 Reduce for independent per-row reductions without alignment requirements and Pattern Reduce for aligned cross-row batch reductions.

# Source material

## 接口选择

| 场景 | 接口 | 参数 | 对齐要求 | 典型用途 |
|-----|------|------|----------|---------|
| 逐行独立处理 | Level 2 | `(dst, src, tmp, count)` | **无** | Softmax, LayerNorm |
| 跨行批量处理 | Pattern | **两种形式**（见下文） | 32 字节 | ReduceSum axis=-1 |

**选择原则**：
- 逐行独立计算 → **Level 2 接口**（更简单，无对齐要求）
- 需要跨行 Reduce → **Pattern 接口**（性能更高，推荐形式1）

---

# Applicability

The structured scope on this Concept is normative.
