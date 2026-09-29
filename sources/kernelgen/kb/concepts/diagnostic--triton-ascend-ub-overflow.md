---
schema_version: '1.0'
id: kg:diagnostic:triton-ascend-ub-overflow
kind: diagnostic
title: Diagnose Triton Ascend UB overflow from the live working set
summary: Route UB overflow checks through interface overhead, live intermediates,
  data type and tile size, and control-flow complexity before reducing tiles or splitting
  the computation.
claim_key: diagnostic.triton_ascend.ub_overflow
domains:
- compiler
- diagnosis
- memory
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:27:53.003737Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/debug_guide/ub_overflow.md#Common Causes and Solutions
  title: Diagnose Triton Ascend UB overflow from the live working set
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: triton
      compiler: triton-ascend
  numerics:
    exact: false
retrieval:
  phases:
  - post_error
  - post_profile
  tasks:
  - diagnosis
  - next_experiment
  symptoms:
  - ub_overflow
  - compilation_failure
  techniques:
  - live_range_reduction
  - subblock_tiling
  - kernel_split
  keywords:
  - UB overflow
  - intermediate variables
  - dtype
  - control flow
  - tile
evidence_state: source_supported
managed:
  content_hash: sha256:c473ca5b58db8b212c33268d7ed848011dbd1486f22b78b853bd17627912e331
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:53.003737Z'
  updated_at: '2026-07-27T07:27:53.003737Z'
---

# Claim

Route UB overflow checks through interface overhead, live intermediates, data type and tile size, and control-flow complexity before reducing tiles or splitting the computation.

# Source material

## Common Causes and Solutions

### 1. Using Interface Parameters That Increase UB Overhead

Certain interfaces automatically add additional processing logic under specific parameter configurations, resulting in increased UB space usage:

#### `propagate_nan` Parameter for `tl.maximum`, `tl.minimum`, and `tl.clamp` Interfaces

**Issue Description:**
When setting `propagate_nan=tl.PropagateNAN.NONE`, the system automatically adds NaN value detection and processing logic.

**Impact:**

- Significantly increases UB space usage
- May cause performance degradation

**Solutions:**

- If input data does not contain NaN values or strict NaN processing semantics are not required, consider adjusting the `propagate_nan` parameter value
- In scenarios with limited UB space, prioritize parameter configurations that do not trigger additional NaN processing

### 2. Excessive Intermediate Variables

**Problem:**
The kernel defines a large number of temporary tensors or intermediate computation results.

**Solutions:**

- Reduce unnecessary intermediate variables
- Reuse allocated buffers
- Split large computations into multiple smaller kernels

### 3. Large Data Types and Shapes

**Problem:**
Using larger data types such as fp64, bf16, or processing high-dimensional/large shape tensors.

**Solutions:**

- Consider splitting large tensors into blocks for processing
- Modify blocking strategies to reduce the size of each block
- Use smaller data types (e.g., fp16 instead of fp32) while meeting precision requirements

### 4. Complex Control Flow or Loops

**Problem:**
The kernel contains complex conditional statements or multi-level nested loops.

**Solutions:**

- Simplify control flow logic
- Reduce loop nesting levels or iteration counts
- Split complex logic into multiple kernels

# Applicability

The structured scope on this Concept is normative.
