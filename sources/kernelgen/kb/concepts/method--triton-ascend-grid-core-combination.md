---
schema_version: '1.0'
id: kg:method:triton-ascend-grid-core-combination
kind: method
title: Combine excess Triton grid work inside physical Ascend cores
summary: When logical grid work exceeds physical NPU cores, combine tasks and execute
  multiple logical blocks inside each core.
claim_key: method.triton_ascend.grid_core_combination
domains:
- optimization
- scheduling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T05:54:19.817001Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/migration_guide/performance_guidelines.md#Combining Grid Cores
  title: Combine excess Triton grid work inside physical Ascend cores
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
  - plateau
  tasks:
  - architecture_selection
  - diagnosis
  - next_experiment
  symptoms:
  - excessive_grid
  - scheduling_overhead
  techniques:
  - grid_core_combination
  keywords:
  - grid
  - core
  - combine
  - block
  - scheduling
evidence_state: source_supported
managed:
  content_hash: sha256:029e6858c43c1fc963bf15f766ccdc5b81a3590711f79052d7326e1fc4905f6f
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T05:54:19.817001Z'
  updated_at: '2026-07-27T05:54:19.817001Z'
---

# Claim

When logical grid work exceeds physical NPU cores, combine tasks and execute multiple logical blocks inside each core.

# Source material

## Combining Grid Cores

### I. Principles for Automatically Combining Grid Cores

Some scenarios requiring migration of Triton operators from GPUs to NPUs. Due to architectural differences, the Triton operators developed on GPUs often utilize large grid core counts. When executed on NPUs, these operators cannot be scheduled all at once. Delivering them in batches introduces significant latency and degrades performance. To optimize NPU-based Triton operators, you need to check the grid core counts first. In cases with large grid core counts, set the environment variable *TRITON_ALL_BLOCKS_PARALLEL* to improve operator execution performance.

# Applicability

The structured scope on this Concept is normative.
