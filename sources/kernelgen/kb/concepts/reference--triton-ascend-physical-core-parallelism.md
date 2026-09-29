---
schema_version: '1.0'
id: kg:reference:triton-ascend-physical-core-parallelism
kind: reference
title: Configure Triton Ascend launches to use available NPU cores
summary: Choose the launch core count to utilize available NPU computing cores; underutilized
  launches reduce parallelism and throughput.
claim_key: reference.triton_ascend.physical_core_parallelism
domains:
- hardware
- language
- programming-model
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T05:54:19.817001Z'
- by: kernelgen/publisher-v1
  at: '2026-07-27T06:13:40.049105Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/migration_guide/architecture_difference.md#Full Utilization of
    Cores
  title: Configure Triton Ascend launches to use available NPU cores
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/migration_guide/architecture_difference.md#Full Utilization of
    Cores
  title: Triton Ascend should map logical work onto physical NPU cores
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
  - initial
  - post_profile
  tasks:
  - constraint_check
  - architecture_selection
  - diagnosis
  symptoms:
  - low_core_utilization
  - excessive_grid
  techniques:
  - core_partition
  keywords:
  - ascend
  - triton
  - grid
  - physical core
  - logical task
evidence_state: source_supported
managed:
  content_hash: sha256:326f62fe132295440e0828c217574fdea7dab30385e86ace99eb4f145d702364
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T05:54:19.817001Z'
  updated_at: '2026-07-27T06:13:40.049105Z'
---

# Claim

Choose the launch core count to utilize available NPU computing cores; underutilized launches reduce parallelism and throughput.

# Source material

### Full Utilization of Cores

Ascend NPUs have multiple computing cores. Properly allocating and fully utilizing all available cores is one of the key factors to improve operator performance.
When calling Triton kernel functions, you can set the **launch** parameter to control the number of cores in use. Take the GELU operator as an example:

```Python
triton_gelu[n, 1, 1](...)  # The first parameter indicates the number of cores in use. n indicates that n cores are in use.
```

By optimizing the number of cores, you can fully schedule and utilize all computing resources, thereby maximizing the degree of parallelism (DOP) and throughput. Without `auto-blockify` (see below), the number of cores in the launched grid must be less than or equal to 65,535.

# Applicability

The structured scope on this Concept is normative.
