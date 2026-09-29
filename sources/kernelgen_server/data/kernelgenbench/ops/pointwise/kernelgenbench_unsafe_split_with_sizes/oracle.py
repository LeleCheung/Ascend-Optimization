REFERENCE_DEVICE = 'target'

import torch

def run(x, split_sizes, dim):
    return torch.ops.aten.unsafe_split_with_sizes(x, split_sizes, dim)
