---
schema_version: '1.0'
id: kg:method:ascendc-buffer-type-selection
kind: method
title: Select TQue for transfer pipelines and TBuf for compute-only storage
summary: Use TQue with EnQue and DeQue for MTE transfer buffers, TBuf for compute-only
  VECCALC storage, and configure double buffering through InitBuffer rather than the
  queue depth.
claim_key: method.ascendc.buffer_type_selection
domains:
- implementation
- language
- memory
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-buffer.md#TBuf vs TQue 选择
  title: Select TQue for transfer pipelines and TBuf for compute-only storage
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
  - architecture_selection
  - implementation
  - diagnosis
  symptoms:
  - buffer_misuse
  - pipeline_stall
  techniques:
  - buffer_selection
  - queue_management
  keywords:
  - TBuf
  - TQue
  - VECCALC
  - VECIN
  - VECOUT
  - InitBuffer
evidence_state: source_supported
managed:
  content_hash: sha256:be77a08aee17c80f19fcd63a25de5c5391776adbb4cba04bc106d9dc28d968dd
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Use TQue with EnQue and DeQue for MTE transfer buffers, TBuf for compute-only VECCALC storage, and configure double buffering through InitBuffer rather than the queue depth.

# Source material

## TBuf vs TQue 选择

| 场景 | 推荐类型 | 说明 |
|-----|---------|------|
| MTE2/MTE3 搬运缓冲区 | `TQue<VECIN/VECOUT>` | 需要与 Vector 并行，需要 EnQue/DeQue |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | 不涉及 MTE 搬运，用 `Get<T>()` 获取 |
| Double Buffer | `TQue` + `InitBuffer(que, 2, size)` | 在 InitBuffer 中设置 num=2 开启 |

---

# Applicability

The structured scope on this Concept is normative.
