REFERENCE_DEVICE = 'target'

import torch

def run(x, other, alpha):
    return torch.ops.aten.sub(x, other, alpha=alpha)
