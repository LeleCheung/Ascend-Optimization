REFERENCE_DEVICE = 'target'

import torch
def run(input, weight, eps):
    x = input.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * y.to(input.dtype)
