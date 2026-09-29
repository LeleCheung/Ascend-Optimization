REFERENCE_DEVICE = 'target'

import torch

def run(x, size):
    return torch.ops.aten.expand(x, size)
