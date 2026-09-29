---
schema_version: '1.0'
id: kg:method:triton-ascend-transfer-compute-overlap
kind: method
title: Tile for transfer and compute overlap on Triton Ascend
summary: Design tiles so the next batch can be transferred while the current batch
  computes, allowing multibuffering to form a transfer-compute pipeline.
claim_key: method.triton_ascend.transfer_compute_overlap
domains:
- memory
- optimization
- scheduling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:27:53.003737Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/programming_guide/index.md#Parallel Storage and Computation
  title: Tile for transfer and compute overlap on Triton Ascend
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
  - pipeline_idle
  - memory_latency
  techniques:
  - multibuffer
  - transfer_compute_overlap
  keywords:
  - pipeline
  - multibuffer
  - double buffer
  - transfer
  - compute
evidence_state: source_supported
managed:
  content_hash: sha256:9337bcc81d007e89abea509c53a2c9c29cad8f556f11f6d7c0f9a7c731f6cf3b
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:53.003737Z'
  updated_at: '2026-07-27T07:27:53.003737Z'
---

# Claim

Design tiles so the next batch can be transferred while the current batch computes, allowing multibuffering to form a transfer-compute pipeline.

# Source material

### Parallel Storage and Computation

Triton-Ascend supports two data processing modes: serial storage and computation and parallel storage and computation.

Serial storage and computation: Data is first transferred from the global memory to the on-chip memory, and then the next batch of data is transferred after the computation is complete. This mode has a significant idle waiting time, and the efficiency is low.

Parallel storage and computation: Computing is performed when the first batch of data is transferred to the on-chip memory. Then, the second batch of data is transferred, and the "transfer + compute" pipeline operation is formed, significantly improving the overall throughput.

The key to implementing parallel storage and computation is to properly design the data tiling policy so that the data required for the next phase can be prepared in advance during the compute of the current batch of data, thereby implementing parallelization of data transfer and computing.
 Currently, the compiler is configured with **multiBuffer** set to **True** by default, and the parallel storage and computation are supported by default.

# Applicability

The structured scope on this Concept is normative.
