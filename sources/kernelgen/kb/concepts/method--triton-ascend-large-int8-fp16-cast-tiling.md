---
schema_version: '1.0'
id: kg:method:triton-ascend-large-int8-fp16-cast-tiling
kind: method
title: Tile large Triton Ascend int8-to-fp16 casts across cores and UB
summary: Flatten large elementwise casts, assign divisible blocks across available
  cores, and introduce an inner tile only when one block cannot fit in UB.
claim_key: method.triton_ascend.large_int8_fp16_cast_tiling
domains:
- datatype
- optimization
- tiling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:23.810116Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/triton-op-designer/references/cases/elemwise-cast.md#优化：二次切分 + 用满UB
  title: Tile large Triton Ascend int8-to-fp16 casts across cores and UB
scope:
  target:
    level: architecture
    backend: ascend
    architecture: DAV_2201
    software:
      language: triton
      compiler: triton-ascend
  operator:
    motifs:
    - elementwise
    dtypes:
    - int8
    - float16
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
  - low_core_utilization
  - ub_underutilization
  - large_elementwise_overhead
  techniques:
  - flattening
  - core_partition
  - ub_tiling
  keywords:
  - int8
  - fp16
  - cast
  - BLOCK_SIZE
  - TILE_SIZE
  - UB
  - large shape
evidence_state: source_supported
managed:
  content_hash: sha256:c65fbf594a6db2c9917bd8fd1a0a7ff244bf0518800855a0d9e2aeb496be5d77
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:23.810116Z'
  updated_at: '2026-07-27T09:21:23.810116Z'
---

# Claim

Flatten large elementwise casts, assign divisible blocks across available cores, and introduce an inner tile only when one block cannot fit in UB.

# Source material

## 优化：二次切分 + 用满UB

```python
# Triton 内核实现：将BLOCK_SIZE分块，每次搬运TILE_SIZE大小的数据
configs = [
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 65536}), # 核数2048, 性能最优！用满UB且无二次切分
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 32768}), # 核数2048，但UB未用满
    triton.Config({"BLOCK_SIZE": 2097152, "TILE_SIZE": 65536}), # 核数64, 并行度低
    triton.Config({"BLOCK_SIZE": 4194304, "TILE_SIZE": 65536}), # 核数32, 并行度更低
]

# 内核操作：
block_start = pid * BLOCK_SIZE
for i in range(0, BLOCK_SIZE, TILE_SIZE):
    offsets = block_start + tl.arange(0, TILE_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask)
    output_data = tl.cast(input_data, tl.float16)
    tl.store(output_ptr + offsets, output_data, mask=mask)
```

### 优化内容
- triton 内核部分使用for循环，尝试进行二次切分，每次搬运TILE_SIZE大小的数据，提高UB的利用率
- 在一定范围内提高核数，并尝试用满UB
- 核内没有二次切分时性能最优（BLOCK_SIZE = TILE_SIZE = 65536）

### 总结
1. 当数据的shape较大时，为了获得更佳的性能，切分值设置尽量能被shape的大小整除
2. 对于单纯的Elementwise操作，将多根轴的元素展开为一根轴，然后在这根轴上进行切分
3. 将block分配给每个线程块，若UB存不下，可考虑多次切分（二次切分）

# Applicability

The structured scope on this Concept is normative.
