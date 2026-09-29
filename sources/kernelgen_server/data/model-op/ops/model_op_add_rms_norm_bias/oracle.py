REFERENCE_DEVICE = 'target'

import math

import torch


def run(x, residual, weight, bias=None, eps=1e-05):
    """In-place fused add + RMSNorm + bias, FlagGems-vllm PR #700 semantics.

    The delivery ships its own torch golden, which the accuracy test treats as
    the trusted reference for the Triton kernel:

        def _torch_fused_add_rms_norm_with_bias(x, residual, weight, bias, eps):
            x = x + residual
            variance = x.pow(2).mean(-1, keepdim=True)
            hidden_states = x * torch.rsqrt(variance + eps)
            return weight * hidden_states + bias, x

    That golden is followed here, with one deliberate difference in where the
    cast lands. The golden runs in the input dtype throughout, while the kernel
    accumulates in float32 (`x = (x + r).to(tl.float32)`) and casts before
    adding bias:

        y = (x * rrms * w).to(X.dtype.element_ty)
        if HAS_BIAS: y = y + b

    The reference normalizes in float32 and casts once, which is the numerically
    faithful reading of the kernel and stays inside the test's own tolerance
    (atol=1e-2, rtol=1e-3 for float16) against the golden. Keeping the reduction
    in float32 avoids float16 overflow in x.pow(2) for the 4096-wide rows the
    workloads use, which the golden's in-dtype `mean` is prone to.

    Mutation contract, from the wrapper's docstring and its two stores:

        tl.store(R + cols, x)   # residual <- the pre-normalization sum
        tl.store(X + cols, y)   # x        <- the normalized, biased result

    so both inputs are modified and returned as (x, residual) by alias.
    """
    if x.shape != residual.shape:
        raise ValueError(
            f"x and residual shapes must match: {x.shape} vs {residual.shape}"
        )
    normalized_shape = tuple(weight.shape)
    if bias is not None and tuple(bias.shape) != normalized_shape:
        raise ValueError(
            f"bias shape {tuple(bias.shape)} must match weight shape "
            f"{normalized_shape}"
        )
    n_elems = math.prod(normalized_shape)
    dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))

    total = (x + residual).to(torch.float32)
    var = (total * total / n_elems).sum(dim=dims, keepdim=True)
    normed = total * torch.rsqrt(var + eps)
    out = normed * weight.to(torch.float32).view(*normalized_shape)
    if bias is not None:
        out = out + bias.to(torch.float32).view(*normalized_shape)

    residual.copy_(total.to(residual.dtype))
    x.copy_(out.to(x.dtype))
    return x, residual
