---
schema_version: '1.0'
id: kg:method:triton-ascend-data-type-optimization
kind: method
title: Select vector-friendly data types in Triton Ascend kernels
summary: Avoid unnecessarily wide or poorly vectorized data types when a narrower
  type preserves required numerical behavior.
claim_key: method.triton_ascend.data_type_optimization
domains:
- compiler
- numerics
- optimization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T05:54:19.817001Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/migration_guide/performance_guidelines.md#Optimizing Data Types
  title: Select vector-friendly data types in Triton Ascend kernels
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
  - constraint_check
  - diagnosis
  - next_experiment
  symptoms:
  - scalar_fallback
  - low_vector_utilization
  techniques:
  - data_type_optimization
  keywords:
  - dtype
  - int64
  - int32
  - fp32
  - vectorization
evidence_state: source_supported
managed:
  content_hash: sha256:5a63168a69a7a8718398caae034c5ed632628e0d99286bb29573bfd1e15c11a2
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T05:54:19.817001Z'
  updated_at: '2026-07-27T05:54:19.817001Z'
---

# Claim

Avoid unnecessarily wide or poorly vectorized data types when a narrower type preserves required numerical behavior.

# Source material

## Optimizing Data Types

### I. Core Principles of Data Type Optimization

Some operations of the A2/A3 vector units do not support certain data types. In this case, the corresponding vector operations will degrade to scalar operations, affecting performance. If the overall operator accuracy is not affected, it is advisable to use supported data types to improve performance.
The following operations are involved.

|  **Operator Name** |  **Unsupported Data Type** |
|---|---|
| Vector Add| int64 |
| Vector Cmp| int64/int32 |

### II. Code Examples

- Example code of the Triton operator Vector Add

    For the following Triton operator, when the input tensors `x` and `y` utilize the int64 data type, `x1 + y1` is expanded into a scalar operation, which degrades performance. Provided that computational accuracy remains unaffected, it is advisable to use the int32 data type.

    ``` diff
    @triton.jit
    def npu_vector_add_kernel(
        x,                          # [Tensor] input tensor (1 x col)
        y,                          # [Tensor] input tensor (1 x col)
        z,                          # [Tensor] output tensor (1 x col)
        vector_len: tl.constexpr,   # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(BLOCK_SIZE)
        len_mask = offset < vector_len
        x1 = tl.load(x + offset, mask=len_mask)
        y1 = tl.load(y + offset, mask=len_mask)
        z1 = x1 + y1
        tl.store(z + offset, z1, mask=len_mask)
    ```

- Example code of the Triton operator Vector Cmp

    In the following Triton operator, the `mask` operation utilizes Cmp. However, Cmp does not support the int64 or int32 data type, causing the condition `cols < N` to be expanded into a scalar operation, which reduces performance. Provided that computational accuracy remains unaffected, it is advisable to use the FP32 data type.
    In Triton operator programming, `mask` operations are frequently employed in syntax such as `load`, `store`, and `where`. During performance optimization, you should prioritize identifying performance degradation caused by these operations.

    ``` diff
    @triton.jit
    def npu_vector_cmp_kernel(
        X,                 # [Tensor] input tensor (row x col)
        Out,               # [Tensor] output tensor (row x col)
        Mean,              # [Vector] mean tensor (row, ) of X
        Rstd,              # [Vector] std tensor (row, ) of X
        stride_x_row,      # [Scalar] stride of row of x
        stride_out_row,    # [Scalar] stride of row of out, normally equals to stride_x_row
        M,                 # [Scalar] row number
        N,                 # [Scalar] col number
        eps,               # [Scalar] epsilon to avoid division by zeros
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        group_m = tl.program_id(0)
        group_n = tl.program_id(1)
        row = group_m

        # calculate index & offset
        Mean = Mean + group_n * M
        Rstd = Rstd + group_n * M
        X = X + row * stride_x_row + group_n * N
        Out = Out + row * stride_out_row + group_n * N

        cols = tl.arange(0, BLOCK_N)  # cols is int64
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

        # calculate mean & rstd
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        # [Changed begin]
    -   xbar = tl.where(cols < N, X - mean, 0.0)
    +   cols_cmp = cols.to(tl.float32)
    +   xbar = tl.where(cols_cmp < N, x - mean, 0.0)
        # [Changed end]

        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)

        # calculate Out
        mask = cols < N
        out = (x - mean) * rstd
        tl.store(Out + cols, out, mask=mask)
    ```

# Applicability

The structured scope on this Concept is normative.
