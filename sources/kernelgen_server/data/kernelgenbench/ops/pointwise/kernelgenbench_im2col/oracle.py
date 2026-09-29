REFERENCE_DEVICE = 'target'

import torch

def run(x, kernel_size, dilation, padding, stride):
    return torch.ops.aten.im2col(x, kernel_size, dilation, padding, stride)
