---
schema_version: '1.0'
id: kg:method:ascendc-enque-deque-pipeline-synchronization
kind: method
title: Synchronize AscendC transfer and compute stages with EnQue and DeQue
summary: Place EnQue and DeQue between asynchronous MTE transfers and vector compute
  so consumers wait for buffer readiness without globally stalling the pipeline.
claim_key: method.ascendc.enque_deque_pipeline_synchronization
domains:
- correctness
- implementation
- scheduling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-pipeline.md#方案一：EnQue/DeQue
    队列同步（推荐）
  title: Synchronize AscendC transfer and compute stages with EnQue and DeQue
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
  - initial
  - post_error
  - post_profile
  tasks:
  - implementation
  - diagnosis
  - next_experiment
  symptoms:
  - random_output
  - missing_synchronization
  - pipeline_stall
  techniques:
  - EnQue
  - DeQue
  - queue_synchronization
  keywords:
  - EnQue
  - DeQue
  - MTE2
  - MTE3
  - asynchronous DMA
  - synchronization
evidence_state: source_supported
managed:
  content_hash: sha256:008f8218f517810707d201ed39a07bd15fcb306720b7d7dee7e7ca0c3f0cffdc
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Place EnQue and DeQue between asynchronous MTE transfers and vector compute so consumers wait for buffer readiness without globally stalling the pipeline.

# Source material

### 方案一：EnQue/DeQue 队列同步（推荐）

**原理**：TQue 的 EnQue/DeQue 机制自动提供硬件同步点。

```cpp
// ✅ 正确：使用 EnQue/DeQue 同步
// Step 1: CopyIn - MTE2 搬运
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
inQueueX.EnQue(xLocal);                    // 标记"就绪"

// Step 2: Compute - Vector 计算
AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 阻塞等待 MTE2 完成
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
AscendC::Adds<float>(yLocal, xIn, 1.0f, count);
outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xIn);

// Step 3: CopyOut - MTE3 搬运
AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 阻塞等待 Vector 完成
AscendC::DataCopyPad(yGm[gmOffset], yOut, copyOutParams);
outQueueY.FreeTensor(yOut);
```

**关键点**：
- `EnQue(xLocal)` 标记 buffer 数据就绪
- `DeQue<float>()` 阻塞等待数据就绪
- DeQue 返回后，数据一定已经搬运完成

# Applicability

The structured scope on this Concept is normative.
