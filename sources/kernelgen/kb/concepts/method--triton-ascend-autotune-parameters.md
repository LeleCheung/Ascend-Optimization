---
schema_version: '1.0'
id: kg:method:triton-ascend-autotune-parameters
kind: method
title: Autotune shape-dependent Triton Ascend block parameters
summary: Benchmark a bounded set of constexpr tiling configurations keyed by workload
  dimensions and cache the best configuration for later calls.
claim_key: method.triton_ascend.autotune_parameters
domains:
- autotuning
- optimization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:27:53.003737Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/programming_guide/index.md#Triton Autotune
  title: Autotune shape-dependent Triton Ascend block parameters
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
  - plateau
  tasks:
  - architecture_selection
  - next_experiment
  - implementation
  symptoms:
  - shape_dependent_performance
  techniques:
  - autotune
  - parameter_sweep
  keywords:
  - autotune
  - config
  - constexpr
  - block size
  - workload key
evidence_state: source_supported
managed:
  content_hash: sha256:08965d7a400bb7f7148d08e34346cc61784e3f0aeab8d3838f09ed49357218f2
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:53.003737Z'
  updated_at: '2026-07-27T07:27:53.003737Z'
---

# Claim

Benchmark a bounded set of constexpr tiling configurations keyed by workload dimensions and cache the best configuration for later calls.

# Source material

### Triton Autotune

In tiling block optimization, the values of block parameters such as **BLOCK_SIZE** and **BLOCK_SIZE_SUB** directly affect operator performance. However, manually trying parameter combinations is inefficient and makes it difficult to find the best values. `triton.autotune` is the autotuning utility provided by the Triton framework. It can sweep over preset parameter configurations, compare their performance, and automatically select the best combination. It is a core tool for tiling optimization.

For the recommended Triton-Ascend usage of `configs=[]`, the scope of automatic tiling, see the [Triton-Ascend Autotune Guide](./../autotune_guide.md).

- Core functions
Automatic exploration of the parameter space: Test different values of constexpr block parameters such as **BLOCK_SIZE** and **BLOCK_SIZE_SUB** in batches.
Benchmark-based comparison: Select the optimal parameters for the current hardware based on execution time.
Caching of tuning results: Cache the best configuration after tuning so that later calls can reuse it without tuning again.

- Simple example

    ```diff
    import triton.language as tl

    @triton.autotune(
    configs=[ # List of parameter configurations to be tested. The candidate parameter values must be powers of 2.
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
        ],
        key=['n_elements'], # Tune dimension: input dimension on which the parameter value depends.
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    ```

- Note: You can set the following environment variables to print the optimal parameter information.

    ```diff
    export TRITON_PRINT_AUTOTUNING=1
    ```

# Applicability

The structured scope on this Concept is normative.
