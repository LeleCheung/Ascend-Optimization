---
schema_version: '1.0'
id: kg:diagnostic:ascendc-reduce-temp-buffer-type
kind: diagnostic
title: Diagnose AscendC Level 2 Reduce temporary-buffer type mismatches
summary: For Level 2 Reduce compilation failures, verify that tmpBuffer uses the same
  element type as the source and destination tensors.
claim_key: diagnostic.ascendc.reduce_temp_buffer_type
domains:
- compiler
- diagnosis
- reduction
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-reduce.md#错误1：tmpBuffer 类型不匹配
  title: Diagnose AscendC Level 2 Reduce temporary-buffer type mismatches
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
  tasks:
  - diagnosis
  - implementation
  symptoms:
  - compile_error
  - reduce_template_mismatch
  techniques:
  - buffer_type_check
  keywords:
  - ReduceMax
  - ReduceSum
  - tmpBuffer
  - LocalTensor
  - type mismatch
evidence_state: source_supported
managed:
  content_hash: sha256:83b07405ebc6414f56ef2d0b3b1bd192c251fa052f41dd6d654c6d5ed65cd580
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

For Level 2 Reduce compilation failures, verify that tmpBuffer uses the same element type as the source and destination tensors.

# Source material

### 错误1：tmpBuffer 类型不匹配

```cpp
// ❌ 错误
AscendC::LocalTensor<uint8_t> tmpBuffer = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax(rowTmp, src, tmpBuffer, count);

// ✅ 正确
AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();
AscendC::ReduceMax(rowTmp, src, reduceTmp, count);
```

# Applicability

The structured scope on this Concept is normative.
