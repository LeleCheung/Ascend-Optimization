---
schema_version: '1.0'
id: kg:reference:ascendc-tque-buffer-count-limits
kind: reference
title: Respect event and buffer limits when allocating AscendC TQue objects
summary: TQue allocation is limited by product event resources, and enabling double
  buffering consumes two buffers per queue and reduces the number of queues that can
  be allocated.
claim_key: reference.ascendc.tque_buffer_count_limits
domains:
- constraint
- hardware
- memory
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-buffer.md#TQue Buffer 数量限制
  title: Respect event and buffer limits when allocating AscendC TQue objects
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
  tasks:
  - constraint_check
  - architecture_selection
  - implementation
  - diagnosis
  symptoms:
  - event_resource_exhaustion
  - buffer_allocation_failure
  techniques:
  - resource_budgeting
  keywords:
  - TQue
  - eventID
  - buffer count
  - double buffer
  - Atlas
evidence_state: source_supported
managed:
  content_hash: sha256:c608bb82ee3dcb1c2e0fd0fca06f21bdfc0dc67d151031fd33e1baeb56555adc
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

TQue allocation is limited by product event resources, and enabling double buffering consumes two buffers per queue and reduces the number of queues that can be allocated.

# Source material

### TQue Buffer 数量限制

| 产品系列 | eventID 数量 | 最大 TQue 数量 |
|---------|-------------|---------------|
| Atlas 训练系列 | 4 | 4 |
| Atlas 推理系列 AI Core | 8 | 8 |
| Atlas 推理系列 Vector Core | 8 | 8 |
| Atlas A2/A3 系列 | 8 | 8 |

**注意**：
- 不开启 Double Buffer（num=1）：最多可申请 8 个 TQue
- 开启 Double Buffer（num=2）：每个 TQue 占用 2 个 buffer，最多只能申请 4 个 TQue

```cpp
// 开启 Double Buffer 时，最多只能申请 4 个 TQue
pipe->InitBuffer(que0, 2, size);  // ✅
pipe->InitBuffer(que1, 2, size);  // ✅
pipe->InitBuffer(que2, 2, size);  // ✅
pipe->InitBuffer(que3, 2, size);  // ✅
pipe->InitBuffer(que4, 2, size);  // ❌ 超过限制
```

# Applicability

The structured scope on this Concept is normative.
