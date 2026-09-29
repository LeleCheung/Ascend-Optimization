REFERENCE_DEVICE = 'target'

import math
import torch
def run(t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period):
    half = dim // 2
    idx = torch.arange(half, dtype=torch.float32, device=t.device)
    exponent = -math.log(max_period) * idx / (half - downscale_freq_shift)
    angle = scale * t.float()[:, None] * torch.exp(exponent)[None, :]
    sin, cos = torch.sin(angle), torch.cos(angle)
    return torch.cat([cos, sin] if flip_sin_to_cos else [sin, cos], dim=-1)
