REFERENCE_DEVICE = 'target'

import math

import torch


def run(x1, x2, normalized_shape, weight, eps=1e-05):
    """Add + RMSNorm, FlagGems-vllm Ascend semantics (PR #731).

    Transcribed from add_rms_norm_kernel:

        x = (x1 + x2).to(tl.float32)
        _var_base = (x * x) / N
        var = tl.sum(_var_base)
        rrms = 1 / tl.sqrt(var + eps)
        y = (x * rrms * w).to(Y.dtype.element_ty)

    Two details that matter:
      * the sum is `sum(x*x/N)`, i.e. each term is divided by N before the
        reduction rather than dividing the total afterwards. In float32 the two
        differ only in rounding, but this form is what the kernel computes;
      * the weight is applied while still in float32, with a single cast to the
        input dtype at the very end.

    N is the product of `normalized_shape`, so normalization spans the trailing
    len(normalized_shape) dimensions, not just the last one.

    Cross-checked on npu:0 against torch_npu.npu_add_rms_norm(x1, x2, weight,
    eps)[0]: exact in float32 up to one ulp, and within the per-dtype workload
    tolerances for float16/bfloat16.

    The kernel clamps its row count to 65535 (`M = min(prod(...), 65535)`), which
    is a launch-geometry limit rather than op semantics; the workloads stay below
    it, so the reference normalizes every row.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"Input shapes must match: {x1.shape} vs {x2.shape}")
    normalized = tuple(int(d) for d in normalized_shape)
    n_elems = math.prod(normalized) if normalized else 1
    dims = tuple(range(x1.ndim - len(normalized), x1.ndim))

    x = (x1 + x2).to(torch.float32)
    var = (x * x / n_elems).sum(dim=dims, keepdim=True)
    y = x * torch.rsqrt(var + eps) * weight.to(torch.float32).view(*normalized)
    return y.to(x1.dtype)
