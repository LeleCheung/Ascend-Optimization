---
schema_version: '1.0'
id: kg:method:ascendc-periodic-offset-table-vectorization
kind: method
title: Vectorize periodic AscendC offset-table construction
summary: Generate one scalar base pattern and expand periodic offset or index tables
  with vector Adds instead of issuing one scalar SetValue per element.
claim_key: method.ascendc.periodic_offset_table_vectorization
domains:
- indexing
- optimization
- vectorization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-transpose.md#1.4 偏移表offsetBuff生成：Scalar
    → Vector 指令优化
  title: Vectorize periodic AscendC offset-table construction
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
  - post_profile
  - plateau
  tasks:
  - diagnosis
  - implementation
  - next_experiment
  symptoms:
  - high_scalar_ratio
  - offset_table_overhead
  - low_vector_utilization
  techniques:
  - periodic_table_vectorization
  - scalar_to_vector
  keywords:
  - offsetBuff
  - SetValue
  - Adds
  - periodic table
  - Scalar
  - Vector
evidence_state: source_supported
managed:
  content_hash: sha256:81b43c2b5ec03aa93ad683842f733cdcac3e4f4fa92c5fb564339607dc365f27
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Generate one scalar base pattern and expand periodic offset or index tables with vector Adds instead of issuing one scalar SetValue per element.

# Source material

### 1.4 偏移表offsetBuff生成：Scalar → Vector 指令优化

**问题**：通用实现用 SetValue 逐元素写 offset 表，tileNA × C 次 Scalar 操作。当 tileNA=2048, C=3 时需 6144 次 Scalar 写入，小规模场景下 Scalar 占比可达 90%，成为性能瓶颈。

**关键观察**：偏移表具有周期性结构——每 16 个 p 值为一组，组间差值恒定为 16 × 16 × sizeof(half) = 512 字节：

```
组 0: offset[p*3+0] = (p*16+0)*2,  offset[p*3+1] = (p*16+1)*2,  offset[p*3+2] = (p*16+2)*2   (p=0..15)
组 1: 与组 0 完全相同，仅每个元素 +512
组 2: 与组 0 完全相同，仅每个元素 +1024
...
```

**优化方法**：Scalar 生成基础模式 + Adds 向量指令批量扩展

```
__aicore__ inline void InitOffsetTable()
{
    auto offsetI32 = offsetBuf.Get<int32_t>();
    uint32_t baseCount = 16 * C;
    // Step 1: Scalar SetValue 生成基础模式（仅 16×C 个元素）
    for (uint32_t p = 0; p < 16; ++p) {
        for (uint32_t c = 0; c < C; ++c) {
            offsetI32.SetValue(p * C + c, (p * 16 + c) * sizeof(half));
        }
    }
    // Step 2: Adds 向量指令扩展后续组（每组一次向量操作）
    uint32_t totalGroups = tileNA / 16;
    for (uint32_t g = 1; g < totalGroups; ++g) {
        AscendC::Adds(offsetI32[g * baseCount], offsetI32[0],
                      static_cast<int32_t>(g * 16 * 16 * sizeof(half)), baseCount);
    }
}
```

| 指标            | 优化前（纯 SetValue） | 优化后（Scalar+Adds） |
| ------------- | --------------- | ---------------- |
| Scalar 调用次数   | 6144            | 48               |
| Adds 向量调用次数   | 0               | 127              |
| Scalar ratio  | 90.5%           | 55.1%            |
| VEC ratio     | 11.1%           | 58.7%            |
| Task Duration | 55.6 us         | 15.3 us          |

**适用条件** :

- 偏移表具有等差数列的周期性结构
- baseCount = 16 × C 需满足 Adds 的对齐要求（32 字节，即 baseCount ≥ 8 对 int32）
- C ≤ 16（小通道 transpose 的典型场景）

**通用模式**：任何具有周期性结构的查找表（offset table、index table 等），都可以用「Scalar 生成基础模式 + Adds 向量扩展」的方式优化，将 Scalar 操作从 O(tileNA × C) 降至 O(16 × C)。

***

# Applicability

The structured scope on this Concept is normative.
