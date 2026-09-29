---
schema_version: '1.0'
id: kg:diagnostic:ascendc-missing-queue-synchronization
kind: diagnostic
title: Detect missing AscendC queue synchronization around asynchronous copies
summary: If data is consumed immediately after DataCopy, add queue synchronization
  or temporarily test with PipeBarrier; correctness restored by the barrier indicates
  a missing synchronization edge.
claim_key: diagnostic.ascendc.missing_queue_synchronization
domains:
- correctness
- diagnosis
- scheduling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-pipeline.md#检查缺少 EnQue/DeQue
  title: Detect missing AscendC queue synchronization around asynchronous copies
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
  - post_error
  - post_evaluation
  tasks:
  - diagnosis
  - next_experiment
  symptoms:
  - random_output
  - nondeterministic_output
  - missing_synchronization
  techniques:
  - barrier_bisection
  - queue_synchronization
  keywords:
  - EnQue
  - DeQue
  - PipeBarrier
  - DataCopy
  - synchronization
evidence_state: source_supported
managed:
  content_hash: sha256:ae9e7d07268e45515e449b7bb2a4c407a05f93fffbc8670e589295db1359d5de
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

If data is consumed immediately after DataCopy, add queue synchronization or temporarily test with PipeBarrier; correctness restored by the barrier indicates a missing synchronization edge.

# Source material

### 检查缺少 EnQue/DeQue

```cpp
// ❌ 错误：AllocTensor 后直接用
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
Compute(x);  // 错！可能读到未完成搬运的数据

// ✅ 正确：DeQue 后再计算
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();  // 等待搬运完成
Compute(xIn);
```

# Applicability

The structured scope on this Concept is normative.
