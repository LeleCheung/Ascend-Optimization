REFERENCE_DEVICE = 'target'

import torch

def run(x, src):
    return torch.ops.aten.copy_(x, src)
