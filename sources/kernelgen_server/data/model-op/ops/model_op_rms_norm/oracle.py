REFERENCE_DEVICE = 'target'

import torch


def run(x, weight, eps=1e-06):
    """SGLang RMSNorm semantics (RMSNorm.forward_native, layernorm.py).

    Reference path, verbatim arithmetic from SGLang:

        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = (x * weight).to(orig_dtype)

    Two details that matter for bit-level parity with the fused kernels:
      * the variance and the reciprocal-sqrt are computed in float32 even for
        float16/bfloat16 inputs (NOTE in the SGLang source: fp32 is required for
        numerical stability);
      * `weight` is applied while still in float32 and the cast back to the
        input dtype happens only afterwards, so there is a single rounding step.

    This is the no-residual branch. The residual branch (fused_add_rmsnorm)
    mutates its inputs and has a different ABI, so it is a separate operator and
    is not covered here.
    """
    orig_dtype = x.dtype
    if not x.is_contiguous():
        x = x.contiguous()
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return (xf * weight.to(torch.float32)).to(orig_dtype)
