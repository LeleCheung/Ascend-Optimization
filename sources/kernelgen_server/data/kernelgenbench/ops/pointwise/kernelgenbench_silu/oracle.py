REFERENCE_DEVICE = 'target'

import torch

def run(x):
    return torch.ops.aten.silu(x)
