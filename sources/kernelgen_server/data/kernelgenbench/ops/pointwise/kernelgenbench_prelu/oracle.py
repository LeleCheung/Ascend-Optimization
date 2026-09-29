REFERENCE_DEVICE = 'target'

import torch

def run(x, weight):
    return torch.ops.aten.prelu(x, weight)
