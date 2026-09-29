---
schema_version: '1.0'
id: kg:method:triton-ascend-int8-matmul-int32-accumulator
kind: method
title: Implement Triton Ascend int8 matmul with an int32 accumulator
summary: Load tiled int8 operands, accumulate tl.dot into an int32 tensor, and convert
  only after the complete K reduction before writing the requested output type.
claim_key: method.triton_ascend.int8_matmul_int32_accumulator
domains:
- implementation
- matrix-multiply
- quantization
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T09:21:23.810116Z'
sources:
- resource: source:cannbot-skills.7fedd2ff5e1f
  locator: ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/02-Core-API/04-linear-algebra-ops.md#进阶用法：int8
    量化矩阵乘法与 dot_scaled
  title: Implement Triton Ascend int8 matmul with an int32 accumulator
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: triton
      compiler: triton-ascend
  operator:
    motifs:
    - matrix_multiply
    dtypes:
    - int8
  numerics:
    exact: false
retrieval:
  phases:
  - initial
  - post_error
  - post_profile
  tasks:
  - architecture_selection
  - implementation
  - diagnosis
  - next_experiment
  symptoms:
  - accumulator_mismatch
  - integer_overflow
  - matmul_wrong_result
  techniques:
  - int32_accumulation
  - tiled_matmul
  keywords:
  - int8
  - matmul
  - tl.dot
  - int32
  - accumulator
  - BLOCK_K
evidence_state: source_supported
managed:
  content_hash: sha256:28a291a353ca58b41b1c9281c7611e5fe45152a38146485a0a0fd23cf10c2a8a
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T09:21:23.810116Z'
  updated_at: '2026-07-27T09:21:23.810116Z'
---

# Claim

Load tiled int8 operands, accumulate tl.dot into an int32 tensor, and convert only after the complete K reduction before writing the requested output type.

# Source material

### 进阶用法：int8 量化矩阵乘法与 dot_scaled

```python
@triton.jit
def int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                     mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        offs_k = offs_k + BLOCK_K

    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

---

# Applicability

The structured scope on this Concept is normative.
