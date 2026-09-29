REFERENCE_DEVICE = 'target'

import torch

def run(x):
    return torch.ops.aten.resolve_conj(x)
