REFERENCE_DEVICE = 'target'

import torch

def run(theta, size, align_corners):
    return torch.ops.aten.affine_grid_generator(theta, size, align_corners)
