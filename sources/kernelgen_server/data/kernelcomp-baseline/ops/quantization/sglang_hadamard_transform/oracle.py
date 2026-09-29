REFERENCE_DEVICE = 'target'

import torch
def _hadamard(n, device, dtype=torch.float32):
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h
def run(x, scale):
    n = x.shape[-1]
    h = _hadamard(n, x.device)
    out = x.float().reshape(-1, n) @ h.t()
    return (out * scale).reshape(x.shape).to(x.dtype)
