---
schema_version: '1.0'
id: kg:diagnostic:ncu-occupancy-interpretation
kind: diagnostic
title: Interpret Nsight Compute occupancy with workload evidence
summary: Low occupancy can reduce latency hiding, but higher occupancy alone does
  not guarantee higher performance; compare achieved and theoretical occupancy with
  other bottleneck evidence.
claim_key: diagnostic.ncu.occupancy_interpretation
domains:
- diagnostics
- hardware
- profiling
status: stable
sources:
- resource: source:nsight-compute-profiling-guide
  locator: occupancy
  title: NVIDIA Nsight Compute Profiling Guide
scope:
  target:
    level: backend
    backend: cuda
  numerics:
    exact: false
retrieval:
  phases:
  - post_profile
  - plateau
  tasks:
  - diagnosis
  - next_experiment
  symptoms:
  - low_occupancy
  - theoretical_achieved_occupancy_gap
  techniques:
  - occupancy_analysis
  keywords:
  - occupancy
  - latency_hiding
  - imbalance
evidence_state: source_supported
managed:
  content_hash: sha256:7ef03fad660e306395a2b7751c6b1b8234d214700738789e592c8306befbc488
  created_by: kernelgen/seed-v1
  created_at: '2026-07-26T00:00:00Z'
  updated_at: '2026-07-26T00:00:00Z'
---

# Diagnostic

Treat occupancy as one diagnostic dimension. Low occupancy limits the number of
warps available to hide latency. A large theoretical-versus-achieved gap can
indicate imbalance. High occupancy is not sufficient evidence that a kernel is
fast.

# Next check

Correlate occupancy with scheduler, memory, instruction, and workload-level
measurements. Form one falsifiable experiment targeting the supported cause,
not an unconditional block-size sweep.
