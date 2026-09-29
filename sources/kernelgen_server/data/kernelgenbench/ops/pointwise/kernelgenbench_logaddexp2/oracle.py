REFERENCE_DEVICE = 'target'

import torch

def run(x, other):
    return torch.ops.aten.logaddexp2(x, other)
