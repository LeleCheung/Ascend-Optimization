---
schema_version: '1.0'
id: kg:method:triton-ascend-ub-staged-discrete-access
kind: method
title: Stage bounded lookup data in UB before discrete selection
summary: When a bounded source fits on chip, load it contiguously into UB and select
  with gather instead of issuing many small discrete global-memory transfers.
claim_key: method.triton_ascend.ub_staged_discrete_access
domains:
- memory
- optimization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:27:53.003737Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/programming_guide/index.md#Transferring Data to the UB and Then
    Selecting the Target Value from the UB
  title: Stage bounded lookup data in UB before discrete selection
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: triton
      compiler: triton-ascend
  operator:
    motifs:
    - gather
    - lookup
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
  - discrete_memory_access
  - low_mte2_efficiency
  techniques:
  - ub_staging
  - gather
  keywords:
  - UB
  - gather
  - lookup
  - random access
  - MTE2
evidence_state: source_supported
managed:
  content_hash: sha256:abbf886fd7b6404c18572177575e2bc0ed350a037f45930e53c3b551cbc3f132
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:53.003737Z'
  updated_at: '2026-07-27T07:27:53.003737Z'
---

# Claim

When a bounded source fits on chip, load it contiguously into UB and select with gather instead of issuing many small discrete global-memory transfers.

# Source material

### Transferring Data to the UB and Then Selecting the Target Value from the UB

[Description] In the discrete scenario of the NPU, data can be transferred to the UB and then the target value can be selected from **share**.

- Example

```diff
@triton.jit
def pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # the original code
    # val = tl.load(x_ptr + idx * stride_x, mask=mask)
    # the modified code
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

- Performance analysis and comparison before and after optimization

You can use the msProf tool to execute the test case to obtain the **PROF_***\** folder, which contains the **op_summary_***\****.csv** file. This file can be used to analyze the pipeline. Note: *\** indicates the timestamp. For details, see the [performance data collection methods](../debug_guide/profiling.md).

||Op Name|aiv_mte2_time(us)|aiv_mte2_ratio|
|:---- |:--------|:--------|:--------|
|Unoptimized|pick_kernel|0.686|0.008|
|Optimized|pick_kernel|1.041|0.066|

According to the data in the table, the values of **aiv_mte2_time(us)** and **aiv_mte2_ratio** before and after the optimization are greatly different. The optimization solution first transfers most of the data to the UB, reducing the number of times that small batches of data are transferred to the UB through the L2 and the total time for transferring data to the UB through the L2.

# Applicability

The structured scope on this Concept is normative.
