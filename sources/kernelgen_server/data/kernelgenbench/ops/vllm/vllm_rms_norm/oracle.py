REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def run(input, weight, epsilon):
    out = torch.empty_like(input)
    if _custom_ops is not None:
        _custom_ops.rms_norm(out, input, weight, epsilon)
    else:
        orig_dtype = input.dtype
        x = input.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
        out.copy_((x * weight.to(torch.float32)).to(orig_dtype))
    return out
