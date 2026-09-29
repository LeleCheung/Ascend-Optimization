REFERENCE_DEVICE = 'target'

import torch

def run(x, mask, value):
    return torch.ops.aten.masked_fill_.Scalar(x, mask, value)
