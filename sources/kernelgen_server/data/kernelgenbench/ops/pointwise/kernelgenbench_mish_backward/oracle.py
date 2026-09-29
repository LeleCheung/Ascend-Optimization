REFERENCE_DEVICE = 'target'

import torch

def run(grad_output, x):
    return torch.ops.aten.mish_backward(grad_output, x)
