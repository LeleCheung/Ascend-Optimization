REFERENCE_DEVICE = 'target'

import torch
def run(x, dtype):
    return torch.ops.aten.to.dtype(x, getattr(torch, dtype), False, True)
