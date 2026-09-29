REFERENCE_DEVICE = 'target'

import torch
def run(x, scale, shift, eps):
    # LayerNorm without affine, computed in fp32 and rounded once, then the
    # per-batch modulation broadcast across the sequence in bf16.
    ln = torch.nn.functional.layer_norm(x.to(torch.float32), (x.shape[-1],), eps=eps)
    return ln.to(x.dtype) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
