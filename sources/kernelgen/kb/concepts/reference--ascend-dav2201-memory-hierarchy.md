---
schema_version: '1.0'
id: kg:reference:ascend-dav2201-memory-hierarchy
kind: reference
title: DAV_2201 on-chip memory and core topology
summary: DAV_2201 devices such as Ascend 910B expose 512 KB L1, 128 KB L0C and 192
  KB UB per documented architecture parameters, with CubeCore and VectorCore in a
  1:2 relationship.
claim_key: reference.ascend.dav2201_memory_hierarchy
domains:
- hardware
- memory
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:27:58.566728Z'
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:32:29.087070Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/npu-arch/references/npu-hardware-params.md#2.3 DAV_2201 — Ascend910B
    / Ascend910_93 系列
  title: DAV_2201 on-chip memory and core topology
scope:
  target:
    level: architecture
    backend: ascend
    architecture: DAV_2201
    devices:
    - Ascend910B
    - Ascend910_93
    software:
      language: triton
      compiler: triton-ascend
  numerics:
    exact: false
retrieval:
  phases:
  - initial
  - post_error
  - post_profile
  tasks:
  - constraint_check
  - architecture_selection
  - diagnosis
  symptoms:
  - ub_overflow
  - l1_overflow
  - core_underutilization
  techniques:
  - hardware_budgeting
  keywords:
  - DAV_2201
  - Ascend910B
  - UB
  - L1
  - L0C
  - CubeCore
  - VectorCore
evidence_state: source_supported
managed:
  content_hash: sha256:c39b348c3e2643bec4a5f58ecf82dafb117f13320428d9f11ea025b3e02eeb09
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:58.566728Z'
  updated_at: '2026-07-27T07:32:29.087070Z'
---

# Claim

DAV_2201 devices such as Ascend 910B expose 512 KB L1, 128 KB L0C and 192 KB UB per documented architecture parameters, with CubeCore and VectorCore in a 1:2 relationship.

# Source material

### 2.3 DAV_2201 — Ascend910B / Ascend910_93 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 2201 |
| L1 | `l1_size` | 512 KB (524288) |
| L0C | `l0_c_size` | 128 KB (131072) |
| UB | `ub_size` | 192 KB (196608) |
| BT | `bt_size` | 1 KB (1024) |
| 稀疏 | `sparsity` | 1（支持 4:2） |
| 核心类型 | `core_type_list` | `CubeCore,VectorCore` |
| 核间关系 | — | CubeCore : VectorCore = 1 : 2 |

# Applicability

The structured scope on this Concept is normative.
