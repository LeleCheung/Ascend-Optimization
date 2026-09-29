---
schema_version: '1.0'
id: kg:method:triton-ascend-instruction-parallelism
kind: method
title: Increase independent instruction work in Triton Ascend kernels
summary: Expose independent operations and reduce dependency chains so the Ascend
  compiler can schedule more instructions in parallel.
claim_key: method.triton_ascend.instruction_parallelism
domains:
- compiler
- optimization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T05:54:19.817001Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/migration_guide/performance_guidelines.md#Optimizing Instruction
    Parallelism
  title: Increase independent instruction work in Triton Ascend kernels
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
  - post_profile
  - plateau
  tasks:
  - diagnosis
  - next_experiment
  - implementation
  symptoms:
  - pipeline_stall
  - dependency_chain
  techniques:
  - instruction_parallelism
  keywords:
  - instruction
  - parallelism
  - dependency
  - schedule
evidence_state: source_supported
managed:
  content_hash: sha256:88e7ad60a960affa0559b8f156904565b6206109220e425b63a92d2c27c57c94
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T05:54:19.817001Z'
  updated_at: '2026-07-27T05:54:19.817001Z'
---

# Claim

Expose independent operations and reduce dependency chains so the Ascend compiler can schedule more instructions in parallel.

# Source material

## Optimizing Instruction Parallelism

### I. Core Principles of Instruction Parallelism Optimization

When executing Triton operators, NPUs leverage parallel mechanisms such as multi-buffer and instruction parallelism to parallelize data-in, computation, and data-out, thereby enhancing performance. However, in certain scenarios, the multi-buffer mechanism cannot be enabled, which reduces the degree of parallelism (DOP) and degrades operator execution performance. If this issue occurs during performance optimization, consider the following aspects and implement optimizations based on the provided code examples:\

1. Data transfer and computation involve dependencies, which introduce synchronization. The memory transfer engine (MTE) can only be triggered after vector computation completes, resulting in low DOP.\
2. In cases where the operator lacks multiple data loads or a single execution completes without tiling, the multi-buffer mechanism cannot be enabled.\
3. The multi-buffer mechanism requires additional UB space. If the UB space is insufficient during computation, the multi-buffer mechanism cannot be enabled.

### II. Code Examples

- Example 1: Reducing synchronization for higher DOP

    In operator optimization, increasing instruction-level parallelism (DOP) is a critical strategy. In the `tl.load` statement below, when `N` > `M`, the loaded data fills only a portion of the tensor memory space pointed to by `data`. For the remaining unfilled portion, if users do not specify the `other` value, GPUs default to zero-padding. To reduce the adaptation workload of migration, NPUs maintain the same behavior as GPUs. NPUs first use the vector core to set all the memory space pointed to by `data` to a specified value (defaulting to `0` if no `other` value is provided). Subsequently, the MTE2 instruction transfers data to part of the memory space pointed to by `data`. This implementation results in a dependency between MTE2 and vector operations, which limits parallelism and degrades overall performance.

    ```diff
    @triton.jit
    def npu_vector_add_kernel(
        input,                          # [Tensor] input tensor (1 x col)
        output,                         # [Tensor] output tensor (1 x col)
        M: tl.constexpr,                # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        N :tl.constexpr = BLOCK_SIZE
        idx = tl.arange(0, N)
        mask = idx < M
        data = tl.load(input + idx, mask = mask) # Alternatively, specify a value such as other=-1.
    ```

    To increase DOP and enhance performance, when the loaded data fills only a portion of the memory space pointed to by `data`, add `care_padding=False` to the load statement to remove default-value padding, provided that the unfilled portion does not affect subsequent computation results. That is, the preceding operator can be optimized follows:

    ```diff
    @triton.jit
    def npu_vector_add_kernel(
        input,                          # [Tensor] input tensor (1 x col)
        output,                         # [Tensor] output tensor (1 x col)
        M: tl.constexpr,                # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.arange(0, N)
        mask = idx < M
    -   data = tl.load(input + idx, mask = mask) # Alternatively, specify a value such as other=-1.
    +   data = tl.load(input + idx, mask = mask, care_padding=False) # Alternatively, specify a value such as other=-1.
    ```

- Example 2: Using `for` loops in Triton operators to increase tiling and enhance DOP

    In Triton operator programming, `mask` operations are frequently employed in syntax such as `load`, `store`, and `where`. During performance optimization, you should prioritize identifying performance degradation caused by these operations. When the logic within Triton operators executes sequentially in a single pass (Start -> Data-in -> Computation -> Data-out -> End), instructions cannot be parallelized, resulting in low execution efficiency. By introducing `for` loops to increase tiling, you can process data in multiple passes (with each pass handling a reduced data volume), enabling parallel execution of data-in, computation, and data-out. This approach reduces serial waiting time and improves overall performance. Additionally, compared to monolithic (non-tiled) data processing, the use of `for` loops for tiling reduces UB space consumption.
    Note: Mathematical equivalence is an important aspect to consider when you increase data tiling.

    ```diff
    @triton.jit
    def alloc_extend_kernel(
            pre_lens_ptr,
            seq_lens_ptr,
            free_page_ptr,
            out_indices,
            bs_upper: tl.constexpr,
            page_size: tl.constexpr,
            max_num_extend_tokens: tl.constexpr,
    +       BLOCK_SIZE: tl.constexpr = 1024,
    ):
        pid = tl.program_id(0)

        load_offset = tl.arange(0, bs_upper)
        seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
        pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
        extend_lens = seq_lens - pre_lens

        seq_len = tl.load(seq_lens_ptr + pid)
        pre_len = tl.load(pre_lens_ptr + pid)
        extend_len = seq_len - pre_len

        sum_extend_lens = tl.sum(extend_lens)
        output_start_loc = sum_extend_lens - extend_len

        num_pages_after = (seq_lens + page_size - 1) // page_size
        num_pages_before = (pre_lens + page_size - 1) // page_size
        num_new_pages = num_pages_after - num_pages_before

        num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
                pre_len + page_size - 1
        ) // page_size
        sum_num_new_pages = tl.sum(num_new_pages)
        new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

        # Part 2: fill the new full pages
        num_part2 = (
                seq_len // page_size * page_size
                - (pre_len + page_size - 1) // page_size * page_size
        )

    -   # load data at once
    -   offset_many_page = tl.arange(0, max_num_extend_tokens)
    -   page_start = tl.load(
    -       free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    -       mask=offset_many_page < num_part2,
    -   )
    -   tl.store(
    -       out_indices + output_start_loc + offset_many_page,
    -       page_start * page_size + offset_many_page % page_size,
    -       mask=offset_many_page < num_part2,
    -   )

    +   # load data using loop
    +   num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    +   blk_offset = tl.arange(0, BLOCK_SIZE)
    +   for i in range(num_loop):
    +       offset_many_page = blk_offset + i * BLOCK_SIZE
    +       page_start = tl.load(
    +           free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    +           mask=offset_many_page < num_part2,
    +       )
    +       tl.store(
    +           out_indices + output_start_loc + offset_many_page,
    +           page_start * page_size + offset_many_page % page_size,
    +           mask=offset_many_page < num_part2,
    +       )
    ```

# Applicability

The structured scope on this Concept is normative.
