REFERENCE_DEVICE = 'target'

import torch

def run(x, size, stride, storage_offset):
    return torch.ops.aten.as_strided(x, size, stride, storage_offset)
