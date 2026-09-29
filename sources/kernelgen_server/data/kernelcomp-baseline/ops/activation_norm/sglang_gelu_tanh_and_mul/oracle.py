REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F
def run(input):
    d = input.shape[-1] // 2
    x1, x3 = input[..., :d].float(), input[..., d:].float()
    return (F.gelu(x1, approximate="tanh") * x3).to(input.dtype)
