---
schema_version: '1.0'
id: kg:method:triton-ascend-block-size-selection
kind: method
title: Maximize Triton Ascend block size within on-chip memory limits
summary: Choose the largest block that fits all live values and buffering in on-chip
  memory, and benchmark candidate sizes rather than assuming the maximum legal value
  is fastest.
claim_key: method.triton_ascend.block_size_selection
domains:
- memory
- optimization
- tiling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:27:53.003737Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/programming_guide/index.md#Setting the Proper Data Block Size (BLOCK
    SIZE)
  title: Maximize Triton Ascend block size within on-chip memory limits
scope:
  target:
    level: backend
    backend: ascend
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
  - plateau
  tasks:
  - architecture_selection
  - diagnosis
  - next_experiment
  symptoms:
  - ub_overflow
  - low_compute_to_memory_ratio
  techniques:
  - block_size_tuning
  - autotune
  keywords:
  - block size
  - on-chip memory
  - UB
  - tiling
  - autotune
evidence_state: source_supported
managed:
  content_hash: sha256:e59a92565b8da68cd2bab52dc73e1162a455782768415c4f3487fa78b9f6d769
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:53.003737Z'
  updated_at: '2026-07-27T07:27:53.003737Z'
---

# Claim

Choose the largest block that fits all live values and buffering in on-chip memory, and benchmark candidate sizes rather than assuming the maximum legal value is fastest.

# Source material

### Setting the Proper Data Block Size (BLOCK SIZE)

Take **add_kernel** as an example. The variables and operations determine the on-chip memory usage. You can change the value of **BLOCK_SIZE** to adjust the size of the data block in the loop and the size of the intermediate result. If the upper limit is exceeded, the expected usage size is displayed and an error is reported during operator compilation. To achieve the maximum compute-to-memory ratio, **BLOCK_SIZE** needs to be as large as possible without exceeding the on-chip space. You can set different **BLOCK_SIZE** values in advance by using [autotune](../examples/06_autotune_example.md) of Triton-Ascend. The optimal setting is automatically selected during running.

```python
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               out_ptr,
               n,  # Total number of elements.
               BLOCK_SIZE: tl.constexpr,  # Number of block elements.
               ):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        # The block size is BLOCK_SIZE.
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        # Load data of x and y to the on-chip memory.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(out_ptr + offsets, output, mask=mask)
```

# Applicability

The structured scope on this Concept is normative.
