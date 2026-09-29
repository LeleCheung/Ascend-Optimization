---
schema_version: '1.0'
id: kg:method:ascendc-cast-round-mode-selection
kind: method
title: Select AscendC Cast RoundMode from the conversion direction
summary: Use CAST_NONE for lossless widening conversions and an explicit rounding
  mode for narrowing or quantizing conversions.
claim_key: method.ascendc.cast_round_mode_selection
domains:
- correctness
- implementation
- numerics
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-precision.md#Cast RoundMode
    选择
  title: Select AscendC Cast RoundMode from the conversion direction
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
  - post_evaluation
  tasks:
  - constraint_check
  - implementation
  - diagnosis
  symptoms:
  - precision_failure
  - cast_mismatch
  techniques:
  - round_mode_selection
  - explicit_cast
  keywords:
  - Cast
  - RoundMode
  - CAST_NONE
  - CAST_ROUND
  - half
  - float
evidence_state: source_supported
managed:
  content_hash: sha256:a12ce3ddbca82967086595c3beb718f092b7e03fc4085bf22b69d57cd1de9a9d
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Use CAST_NONE for lossless widening conversions and an explicit rounding mode for narrowing or quantizing conversions.

# Source material

## Cast RoundMode 选择

### 选择规则

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | 低精度→高精度，无精度损失 |
| **float → half** | `CAST_ROUND` | 高精度→低精度，有精度损失 |
| half → int32_t | `CAST_ROUND` / `CAST_CEIL` | 量化场景，根据需求选择 |
| int32_t → float | `CAST_NONE` | 整数→浮点，无精度损失 |

### 正确用法

```cpp
// ✅ half → float：低精度到高精度
AscendC::LocalTensor<float> xFloat = workBuf.Get<float>();
AscendC::Cast<float, half>(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, count);

// ✅ float → half：高精度到低精度
AscendC::LocalTensor<half> yHalf = outQueue.AllocTensor<half>();
AscendC::Cast<half, float>(yHalf, xFloat, AscendC::RoundMode::CAST_ROUND, count);
```

---

# Applicability

The structured scope on this Concept is normative.
