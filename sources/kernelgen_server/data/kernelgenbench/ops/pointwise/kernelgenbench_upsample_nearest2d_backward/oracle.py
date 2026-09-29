REFERENCE_DEVICE = 'target'

import torch
def run(grad_output, output_size, input_size, scales_h, scales_w):
    return torch.ops.aten.upsample_nearest2d_backward(
        grad_output, output_size, input_size, scales_h, scales_w
    )
