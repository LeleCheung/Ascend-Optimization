REFERENCE_DEVICE = 'target'

import torch

def run(x):
    return torch.ops.aten.square(x)
