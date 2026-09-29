REFERENCE_DEVICE = 'target'

import torch

def run(x, other):
    return torch.ops.aten.le(x, other)
