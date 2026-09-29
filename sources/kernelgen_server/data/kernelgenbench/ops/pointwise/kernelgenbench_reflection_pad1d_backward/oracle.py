REFERENCE_DEVICE = 'target'

import torch

def run(grad_output, x, padding):
    return torch.ops.aten.reflection_pad1d_backward(grad_output, x, padding)
