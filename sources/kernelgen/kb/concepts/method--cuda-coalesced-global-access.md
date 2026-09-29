---
schema_version: '1.0'
id: kg:method:cuda-coalesced-global-access
kind: method
title: Design CUDA global-memory access for coalescing
summary: Map neighboring lanes to nearby addresses so warp memory requests require
  as few global-memory transactions as practical.
claim_key: method.cuda.coalesced_global_access
domains:
- hardware
- memory
- optimization
status: stable
sources:
- resource: source:cuda-best-practices
  locator: coalesced-access-to-global-memory
  title: CUDA C++ Best Practices Guide
scope:
  target:
    level: backend
    backend: cuda
  numerics:
    exact: false
retrieval:
  phases:
  - initial
  - post_profile
  - plateau
  tasks:
  - architecture_selection
  - implementation
  - next_experiment
  symptoms:
  - memory_bound
  - excessive_global_transactions
  techniques:
  - coalesced_global_access
  keywords:
  - coalescing
  - bandwidth
  - stride
evidence_state: source_supported
managed:
  content_hash: sha256:793bef5a504879934ce9d786962787391adc57c2cb8932e6004da2b912a8f647
  created_by: kernelgen/seed-v1
  created_at: '2026-07-26T00:00:00Z'
  updated_at: '2026-07-26T00:00:00Z'
---

# Method

Inspect how consecutive lanes map to global addresses. Prefer contiguous,
aligned access patterns; when the logical layout is strided, consider changing
tile ownership or staging data before adding more arithmetic parallelism.

# Validation

Run the full correctness and performance evaluation. When profiler support is
available, compare global-memory transaction and throughput evidence on the
affected workloads. Do not assume coalescing alone proves a speedup.
