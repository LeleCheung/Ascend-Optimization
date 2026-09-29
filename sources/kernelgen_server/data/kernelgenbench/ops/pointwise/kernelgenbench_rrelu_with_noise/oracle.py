REFERENCE_DEVICE = 'target'

import torch
def run(x, noise, lower, upper, training, generator_seed):
    generator = torch.Generator(device=x.device).manual_seed(generator_seed) if training else None
    return torch.ops.aten.rrelu_with_noise(x, noise, lower, upper, training, generator)
