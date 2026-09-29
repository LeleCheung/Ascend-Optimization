---
schema_version: '1.0'
id: kg:reference:ascendc-transpose-vecout-depth
kind: reference
title: Keep AscendC small-channel transpose VECOUT depth at least two
summary: Multi-tile small-channel transpose requires at least two VECOUT queue slots
  so CopyOut can overlap later compute without stalling the pipeline.
claim_key: reference.ascendc.transpose_vecout_depth
domains:
- constraint
- language
- scheduling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-transpose.md#2.3 `VECOUT`
    depth 必须 >= 2
  title: Keep AscendC small-channel transpose VECOUT depth at least two
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
  - post_profile
  tasks:
  - constraint_check
  - architecture_selection
  - implementation
  - diagnosis
  symptoms:
  - pipeline_deadlock
  - multi_tile_stall
  techniques:
  - queue_depth_configuration
  keywords:
  - VECOUT
  - depth
  - transpose
  - CopyOut
  - pipeline
evidence_state: source_supported
managed:
  content_hash: sha256:ddbc5f0bf671e1867498dcfd45c1704c611b03422001640f262ebeffa34a1ba0
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Multi-tile small-channel transpose requires at least two VECOUT queue slots so CopyOut can overlap later compute without stalling the pipeline.

# Source material

### 2.3 `VECOUT` depth 必须 >= 2

即便 `Compute` 逻辑看起来是“算完立刻写”，也不要把 `VECOUT` 队列缩成 1。多 tile 下 CopyOut 与后续 Compute 交错时，单槽位容易卡死流水。

# Applicability

The structured scope on this Concept is normative.
