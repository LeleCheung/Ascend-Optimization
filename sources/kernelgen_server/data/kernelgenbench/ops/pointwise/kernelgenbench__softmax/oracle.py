REFERENCE_DEVICE = 'target'

import torch

def run(x, dim, half_to_float):
    return torch.ops.aten._softmax(x, dim, half_to_float)
