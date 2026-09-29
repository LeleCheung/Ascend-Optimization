REFERENCE_DEVICE = 'target'

import torch
def run(x, weight, scale, shift, eps):
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    # The norm and the weight multiply happen in fp32 and round **once**, at
    # the end of the norm; the modulation that follows is bf16 arithmetic.
    # This ordering is what makes the baseline bit-exact vs the eager chain.
    normed = (xf * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(x.dtype)
    # scale/shift are [B, 1, D]: one modulation vector per batch element,
    # broadcast across the sequence.
    return normed * (1 + scale) + shift
