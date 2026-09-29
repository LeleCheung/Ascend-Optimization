---
schema_version: '1.0'
id: kg:method:ascendc-vector-repeat-batching
kind: method
title: Batch AscendC Vector API repeats that exceed the encoded range
summary: Split row or tile repetitions into batches no larger than the API repeatTime
  limit and advance tensor offsets between batches.
claim_key: method.ascendc.vector_repeat_batching
domains:
- correctness
- implementation
- tiling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T08:57:00.054353Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/ascendc-api-best-practices/references/api-repeat-limits.md#方案二：Kernel
    侧分批处理
  title: Batch AscendC Vector API repeats that exceed the encoded range
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
  - architecture_selection
  - implementation
  - diagnosis
  - next_experiment
  symptoms:
  - repeat_overflow
  - large_row_count
  - boundary_256_failure
  techniques:
  - repeat_batching
  - tiled_execution
  keywords:
  - repeatTime
  - batch
  - MAX_REPEAT
  - rowOffset
  - '255'
evidence_state: source_supported
managed:
  content_hash: sha256:d16bee6b13b038d3e75460146ce659ead173ae1f0fa3b06f64a285ee6123fe08
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T08:57:00.054353Z'
  updated_at: '2026-07-27T08:57:00.054353Z'
---

# Claim

Split row or tile repetitions into batches no larger than the API repeatTime limit and advance tensor offsets between batches.

# Source material

### 方案二：Kernel 侧分批处理

```cpp
void SubWithBroadcast(
    AscendC::LocalTensor<float>& dst,
    AscendC::LocalTensor<float>& src0,
    AscendC::LocalTensor<float>& src1,
    uint32_t a0Count,
    uint32_t alignedCols,
    uint32_t rowCount)
{
    constexpr uint32_t MAX_REPEAT = 255;
    uint32_t repStride = alignedCols / BLOCK_ELEMENTS;  // BLOCK_ELEMENTS = 8
    
    for (uint32_t col = 0; col < a0Count; col += MASK_FP32) {
        uint32_t curMask = std::min(a0Count - col, MASK_FP32);
        
        // 分批处理
        uint32_t processedRows = 0;
        while (processedRows < rowCount) {
            uint32_t batchRepeat = std::min(rowCount - processedRows, MAX_REPEAT);
            uint32_t rowOffset = processedRows * alignedCols;
            
            AscendC::Sub<float>(
                dst[rowOffset + col],
                src0[rowOffset + col],
                src1[col],
                curMask,
                batchRepeat,
                {1, 1, 1, static_cast<uint8_t>(repStride), static_cast<uint8_t>(repStride), 0}
            );
            
            processedRows += batchRepeat;
        }
    }
}
```

---

# Applicability

The structured scope on this Concept is normative.
