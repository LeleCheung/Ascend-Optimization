---
schema_version: '1.0'
id: kg:reference:triton-ascend-integer-exact-comparison
kind: reference
title: Validate Triton Ascend integer outputs with exact equality
summary: Compare int8 and other integer outputs with exact element equality rather
  than floating-point tolerances, after moving both results to a comparable device.
claim_key: reference.triton_ascend.integer_exact_comparison
domains:
- correctness
- numerics
- validation
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:14.771517Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/debug_guide/precision.md#Integer Types
  title: Validate Triton Ascend integer outputs with exact equality
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: triton
      compiler: triton-ascend
  operator:
    dtypes:
    - int8
    - int16
    - int32
    - int64
  numerics:
    exact: true
retrieval:
  phases:
  - post_evaluation
  - post_error
  tasks:
  - constraint_check
  - diagnosis
  symptoms:
  - integer_mismatch
  - precision_failure
  techniques:
  - exact_comparison
  - torch_equal
  keywords:
  - int8
  - integer
  - exact equality
  - torch.equal
  - precision
evidence_state: source_supported
managed:
  content_hash: sha256:a2869b53c7aeb9e9fdbc538d192ca709bb53834dacafb1f3f7c711556a3be3b1
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:14.771517Z'
  updated_at: '2026-07-27T09:21:14.771517Z'
---

# Claim

Compare int8 and other integer outputs with exact element equality rather than floating-point tolerances, after moving both results to a comparable device.

# Source material

### Integer Types

Integer and boolean types do not allow any error; they must be strictly identical. When comparing across devices, ensure the data has been moved to the same device (e.g., CPU) to avoid misjudgment caused by underlying representation differences.

# Applicability

The structured scope on this Concept is normative.
