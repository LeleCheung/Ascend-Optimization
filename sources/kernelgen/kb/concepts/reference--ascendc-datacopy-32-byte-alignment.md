---
schema_version: '1.0'
id: kg:reference:ascendc-datacopy-32-byte-alignment
kind: reference
title: AscendC DataCopy requires 32-byte aligned transfer sizes
summary: DataCopy requires data sizes aligned to 32 bytes, corresponding to 16 half,
  8 float or int32, or 32 one-byte elements.
claim_key: reference.ascendc.datacopy_32_byte_alignment
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
  locator: ops/ascendc-api-best-practices/references/api-datacopy.md#32 字节对齐要求
  title: AscendC DataCopy requires 32-byte aligned transfer sizes
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
  - unaligned_transfer
  - data_corruption
  techniques:
  - alignment
  - DataCopyPad
  keywords:
  - DataCopy
  - 32 byte
  - alignment
  - half
  - float
  - fp8
evidence_state: source_supported
managed:
  content_hash: sha256:abe83d45d36c04b12a21734eed3b7a1e7f28e108cb5b61c901f88de22a495832
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

DataCopy requires data sizes aligned to 32 bytes, corresponding to 16 half, 8 float or int32, or 32 one-byte elements.

# Source material

## 32 字节对齐要求

**DataCopy 要求 32 字节对齐**，非对齐会导致数据错误。

| 数据类型 | 对齐元素数 | 最小对齐字节数 |
|---------|-----------|--------------|
| half (2 bytes) | 16 | 32 |
| float (4 bytes) | 8 | 32 |
| int32_t (4 bytes) | 8 | 32 |
| fp8 (1 byte) | 32 | 32 |

---

# Applicability

The structured scope on this Concept is normative.
