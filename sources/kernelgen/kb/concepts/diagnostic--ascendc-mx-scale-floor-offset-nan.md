---
schema_version: '1.0'
id: kg:diagnostic:ascendc-mx-scale-floor-offset-nan
kind: diagnostic
title: Diagnose MX quantization NaNs caused by a floor-biased E8M0 scale
summary: If MX-format casts produce NaN, verify that the E8M0 exponent encoding includes
  the ceil offset so the scaled maximum remains within the target dtype range.
claim_key: diagnostic.ascendc.mx_scale_floor_offset_nan
domains:
- diagnosis
- numerics
- quantization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-precision.md#反模式：floor 偏移导致
    NaN
  title: Diagnose MX quantization NaNs caused by a floor-biased E8M0 scale
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
  - nan_output
  - quantization_overflow
  - mx_scale_error
  techniques:
  - e8m0_scale_check
  - exponent_inspection
  keywords:
  - MX
  - E8M0
  - floor
  - ceil
  - NaN
  - fp8
  - fp4
evidence_state: source_supported
managed:
  content_hash: sha256:ccfc65bc2f52b66c995440f7979c064ec1a2ed98b7b000196de4218a56427d27
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

If MX-format casts produce NaN, verify that the E8M0 exponent encoding includes the ceil offset so the scaled maximum remains within the target dtype range.

# Source material

### 反模式：floor 偏移导致 NaN

```cpp
❌ e8m0_byte = biased_exp_amax - emax_quant_dtype;  // 缺失 +1，落入 floor 区间
```

floor 偏移下 `amax / decoded_scale` 可能落在 `[quant_dtype_max, 2 × quant_dtype_max)` 区间（对 e4m3 即 `[448, 896)`）。Cast<量化 dtype, fp32, RINT> 对超过 dtype_max 的输入会产出 NaN（对 fp8_e4m3 为 `0x7F`）。

# Applicability

The structured scope on this Concept is normative.
