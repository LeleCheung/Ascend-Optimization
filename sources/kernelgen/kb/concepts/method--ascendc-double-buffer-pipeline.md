---
schema_version: '1.0'
id: kg:method:ascendc-double-buffer-pipeline
kind: method
title: Overlap AscendC MTE transfers and vector compute with double buffering
summary: Configure two TQue buffers for MTE2 and MTE3 paths and use a single queue-managed
  tile loop so data transfer can overlap vector computation.
claim_key: method.ascendc.double_buffer_pipeline
domains:
- memory
- optimization
- scheduling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-buffer.md#Double Buffer 流水线并行
  title: Overlap AscendC MTE transfers and vector compute with double buffering
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
  - post_profile
  - plateau
  tasks:
  - architecture_selection
  - implementation
  - diagnosis
  - next_experiment
  symptoms:
  - serialized_pipeline
  - mte_idle
  - vector_idle
  techniques:
  - double_buffer
  - transfer_compute_overlap
  keywords:
  - Double Buffer
  - MTE2
  - MTE3
  - Vector
  - TQue
  - overlap
evidence_state: source_supported
managed:
  content_hash: sha256:19dbfecf67f4383395e28d008042e1b665e1f7d50f9d56f3ad60d97d0f72f9d0
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Configure two TQue buffers for MTE2 and MTE3 paths and use a single queue-managed tile loop so data transfer can overlap vector computation.

# Source material

## Double Buffer 流水线并行

### 核心认知

**Double Buffer 不是"用2块内存计算"，而是"用2块内存做搬入/搬出，使 MTE2/MTE3 与 Vector 计算并行"。**

本质：**内存搬运与计算并行，掩盖搬运延迟**。

### 硬件原理

- **MTE2**：搬运工，GM → UB
- **Vector**：加工员，计算
- **MTE3**：搬运工，UB → GM

### 时间线对比

**无 Double Buffer（串行）**：
```
Row 0: [MTE2][Vector][MTE3]
Row 1:                      [MTE2][Vector][MTE3]
```

**有 Double Buffer（并行）**：
```
Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
                  ↑ MTE2与Vector并行！
```

### 实现原则

| Buffer 类型 | InitBuffer num | 说明 |
|------------|----------------|------|
| `TQue<VECIN>` (MTE2 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TQue<VECOUT>` (MTE3 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TBuf<VECCALC>` (纯计算) | - | TBuf 不涉及 MTE 搬运 |

### 正确用法

```cpp
// 1. Init: num=2 开启 Double Buffer
pipe->InitBuffer(inQueueX,  2, tileSize * sizeof(T));
pipe->InitBuffer(outQueueY, 2, tileSize * sizeof(T));
pipe->InitBuffer(workBuf, workSize * sizeof(T));

// 2. Process: 单循环结构，TQue 自动轮转
for (int i = 0; i < totalTiles; i++) {
    CopyIn(i);   // MTE2 异步搬运
    Compute(i);  // Vector 计算
    CopyOut(i);  // MTE3 异步搬出
}

// 3. CopyIn
void CopyIn(int i) {
    LocalTensor<T> x = inQueueX.AllocTensor<T>();
    DataCopyPad(x, xGm[i * tileSize], {1, (uint32_t)(tileSize * sizeof(T)), 0, 0, 0}, {false, 0, 0, static_cast<T>(0)});
    inQueueX.EnQue(x);
}

// 4. Compute
void Compute(int i) {
    LocalTensor<T> x = inQueueX.DeQue<T>();
    LocalTensor<T> y = outQueueY.AllocTensor<T>();
    Add(y, x, constTensor, tileSize);
    outQueueY.EnQue(y);
    inQueueX.FreeTensor(x);
}

// 5. CopyOut
void CopyOut(int i) {
    LocalTensor<T> y = outQueueY.DeQue<T>();
    DataCopyPad(yGm[i * tileSize], y, {1, (uint32_t)(tileSize * sizeof(T)), 0, 0, 0});
    outQueueY.FreeTensor(y);
}
```

### 为什么能并行？

| 操作 | 特性 |
|------|------|
| `DataCopy` | 异步 DMA，立即返回 |
| `EnQue` | 非阻塞，标记就绪 |
| `DeQue` | 阻塞，等待就绪 |

### 常见误区

| 误区 | 正确理解 |
|------|---------|
| 需要手动拆成 Ping/Pong 两套代码 | 单循环 + `InitBuffer(que, 2, size)` 自动管理 |
| depth 模板参数控制 Double Buffer | Double Buffer 由 `InitBuffer` 的 `num` 参数控制 |
| depth 越大越好 | 模板 depth 通常设为 1，性价比最高 |
| 所有 buffer 都要 num=2 | 只有涉及 MTE 搬运的才需要 Double Buffer |

---

# Applicability

The structured scope on this Concept is normative.
