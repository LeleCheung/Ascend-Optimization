---
schema_version: '1.0'
id: kg:method:triton-ascend-subblock-tiling
kind: method
title: Subdivide Triton Ascend work to fit UB capacity
summary: Split a logical block into looped subblocks when the full live working set
  exceeds UB capacity, preserving masks and minimizing each subblock's live intermediates.
claim_key: method.triton_ascend.subblock_tiling
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
  locator: docs/en/programming_guide/index.md#Tiling Optimization
  title: Subdivide Triton Ascend work to fit UB capacity
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
  - post_error
  - post_profile
  - plateau
  tasks:
  - diagnosis
  - implementation
  - next_experiment
  symptoms:
  - ub_overflow
  - excessive_live_values
  techniques:
  - subblock_tiling
  keywords:
  - UB
  - subblock
  - tiling
  - loop
  - working set
evidence_state: source_supported
managed:
  content_hash: sha256:4900efa64d13d2926727502dc80a617d29afddfd5ecd6eb409c71d49ed8f35ae
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:53.003737Z'
  updated_at: '2026-07-27T07:27:53.003737Z'
---

# Claim

Split a logical block into looped subblocks when the full live working set exceeds UB capacity, preserving masks and minimizing each subblock's live intermediates.

# Source material

### Tiling Optimization

Before the AI Core performs computation, data needs to be transferred to the on-chip memory. The on-chip memory space is usually much smaller than the total data volume to be processed by the AI Core. For example, the on-chip memory capacity of Atlas 800T/I A2 is 192 KB. After doublebuffer is enabled by default, the capacity is reduced to half of the original capacity. Therefore, data needs to be tiled during operator computation, and only a small part of the data is loaded and processed each time.

- Example

```diff
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
- def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
+ def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
+   base_offset = pid * BLOCK_SIZE

+   # Calculate the total number of blocks that need to be processed
+   num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

+   # Loop processing each sub block
+   for sub_block_idx in range(num_sub_blocks):
+       # Calculate the offset of the current sub block
+       sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
+       offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
-       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        # Load input and mask
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # Write the original input first
        tl.store(out + offsets, input_vals, mask=mask)

        # Overlay and write value at the position that needs to be filled
-       value_to_write = tl.full([BLOCK_SIZE], value, dtype=input_vals.dtype)
+       value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0))
        tl.store(out + offsets, overwrite_vals, mask=mask)
```

# Applicability

The structured scope on this Concept is normative.
