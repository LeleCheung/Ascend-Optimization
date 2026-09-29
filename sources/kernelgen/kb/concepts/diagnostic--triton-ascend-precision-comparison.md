---
schema_version: '1.0'
id: kg:diagnostic:triton-ascend-precision-comparison
kind: diagnostic
title: Compare Triton Ascend output against a dtype-aware golden result
summary: Generate an equivalent golden result, run the Ascend kernel on identical
  inputs, and compare with dtype-appropriate approximate or exact equality including
  explicit NaN handling.
claim_key: diagnostic.triton_ascend.precision_comparison
domains:
- diagnosis
- numerics
- validation
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T07:27:53.003737Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/debug_guide/precision.md#1. Precision Comparison Workflow
  title: Compare Triton Ascend output against a dtype-aware golden result
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
  - post_evaluation
  tasks:
  - constraint_check
  - diagnosis
  symptoms:
  - precision_failure
  - dtype_mismatch
  - nan_mismatch
  techniques:
  - golden_comparison
  - dtype_aware_tolerance
  keywords:
  - precision
  - golden
  - assert_close
  - tolerance
  - NaN
evidence_state: source_supported
managed:
  content_hash: sha256:c9dbe69f64e90708b821b3bfae84e7af456de5be1ddb11575e88c06cf04dbee4
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T07:27:53.003737Z'
  updated_at: '2026-07-27T07:27:53.003737Z'
---

# Claim

Generate an equivalent golden result, run the Ascend kernel on identical inputs, and compare with dtype-appropriate approximate or exact equality including explicit NaN handling.

# Source material

## 1. Precision Comparison Workflow

### Basic Steps

1. **Obtain Reference Result (Golden)**: Compute results using equivalent Torch operators on CPU/GPU/NPU, or results from the same Triton operator on CPU/GPU

2. **Obtain Triton Result**: Run the Triton kernel on Ascend NPU to get the computation result

3. **Compare and Evaluate**: Use `torch.testing.assert_close` to determine whether the precision requirements are met

### Example: Vector Add

```python
import torch
import triton
import triton.language as tl


def test_vector_add(n, dtype):
    # 1. Input data
    x = torch.randn(n, dtype=dtype, device="cpu")
    y = torch.randn(n, dtype=dtype, device="cpu")

    # 2. Reference result (PyTorch CPU)
    torch_ref = x + y

    # 3. Triton kernel
    @triton.jit
    def add_kernel(in0_ptr, in1_ptr, out_ptr, n: tl.constexpr):
        idx = tl.arange(0, n)
        a = tl.load(in0_ptr + idx)
        b = tl.load(in1_ptr + idx)
        tl.store(out_ptr + idx, a + b)

    def triton_func(x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x.npu(), y.npu(), out, n=x.numel())
        return out

    triton_cal = triton_func(x, y)

    # 4. Precision comparison
    compare_precision(triton_cal.cpu(), torch_ref)
```

# Applicability

The structured scope on this Concept is normative.
