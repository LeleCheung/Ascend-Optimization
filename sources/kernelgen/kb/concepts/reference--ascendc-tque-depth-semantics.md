---
schema_version: '1.0'
id: kg:reference:ascendc-tque-depth-semantics
kind: reference
title: Distinguish AscendC TQue depth from double-buffer count
summary: TQue depth controls consecutive queue operations and is independent of double
  buffering, which is configured by the InitBuffer num argument.
claim_key: reference.ascendc.tque_depth_semantics
domains:
- constraint
- language
- memory
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-buffer.md#depth 参数关键说明
  title: Distinguish AscendC TQue depth from double-buffer count
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
  - implementation
  - diagnosis
  symptoms:
  - queue_depth_error
  - buffer_misconfiguration
  techniques:
  - queue_management
  keywords:
  - TQue
  - depth
  - double buffer
  - InitBuffer
  - num
evidence_state: source_supported
managed:
  content_hash: sha256:cd4e98c304409beabc997d46c9b2fd1d60612c028b5db9523bb3690cfb533ecb
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

TQue depth controls consecutive queue operations and is independent of double buffering, which is configured by the InitBuffer num argument.

# Source material

### depth 参数关键说明

| depth 值 | 适用场景 | 说明 |
|---------|---------|------|
| `depth=1` | **默认推荐**，非 Tensor 原地操作 | 编译器有特殊优化，性能更好 |
| `depth=0` | **Tensor 原地操作** | 需要设置 |
| `depth=2` | 连续 2 次 EnQue 场景 | 与 InitBuffer 的 num 参数独立 |

**注意**：`depth` 与 Double Buffer 无关。Double Buffer 由 `InitBuffer` 的 `num` 参数控制。

```cpp
// ✅ 非连续入队（普通场景）：depth=1 即可
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);
auto tensor = que.AllocTensor<T>();
que.EnQue(tensor);
tensor = que.DeQue<T>();
que.FreeTensor(tensor);
```

# Applicability

The structured scope on this Concept is normative.
