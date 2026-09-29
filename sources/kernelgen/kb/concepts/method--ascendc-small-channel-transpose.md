---
schema_version: '1.0'
id: kg:method:ascendc-small-channel-transpose
kind: method
title: Implement small-channel AscendC transpose with TransDataTo5HD and Gather
summary: Transpose padded 16-row blocks with TransDataTo5HD and use Gather to remove
  padded channels when the input channel count is smaller than 16.
claim_key: method.ascendc.small_channel_transpose
domains:
- implementation
- layout
- vectorization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-transpose.md#1.1 用 `TransDataTo5HD
    + Gather` 做小通道 transpose
  title: Implement small-channel AscendC transpose with TransDataTo5HD and Gather
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: ascendc
  operator:
    motifs:
    - transpose
  numerics:
    exact: false
retrieval:
  phases:
  - initial
  - post_error
  - post_profile
  tasks:
  - architecture_selection
  - implementation
  - diagnosis
  - next_experiment
  symptoms:
  - small_channel_transpose
  - scalar_transpose
  - low_vector_utilization
  techniques:
  - TransDataTo5HD
  - Gather
  - padding
  keywords:
  - transpose
  - small channel
  - TransDataTo5HD
  - Gather
  - 16 rows
evidence_state: source_supported
managed:
  content_hash: sha256:b0bc0b5b56a85ff202d4e0beddb900dc8ec4000f4ff24a714f95bbaf3d2739b2
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Transpose padded 16-row blocks with TransDataTo5HD and use Gather to remove padded channels when the input channel count is smaller than 16.

# Source material

### 1.1 用 `TransDataTo5HD + Gather` 做小通道 transpose

原理说明：

- step1. TransDataTo5HD 每次一定要输入16行(不满16行时也需要填充有效数据,  比如第0行，否则会出现未知异常)，指令操作将16行\[16, N]转为16列\[N, 16]，一次repeat完成\[16,16]的转置，repeat (N + 15) / 16 次后，得到\[N, 16]
- step2. 当转置前的行数\[比如C]小于16时，需要通过Gather操作从前面TransDataTo5HD得到的\[N, 16]，gather出\[N, C]，offset需要在kernel中提前构造好；

# Applicability

The structured scope on this Concept is normative.
