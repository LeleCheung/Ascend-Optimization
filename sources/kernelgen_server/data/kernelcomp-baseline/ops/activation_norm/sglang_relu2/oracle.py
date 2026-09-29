REFERENCE_DEVICE = 'target'

import torch
def run(input):
    x = torch.relu(input.float())
    return (x * x).to(input.dtype)
