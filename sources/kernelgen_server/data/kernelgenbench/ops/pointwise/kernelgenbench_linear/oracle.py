REFERENCE_DEVICE = 'target'

import torch

def run(input, weight, bias):
    return torch.ops.aten.linear(input, weight, bias)
