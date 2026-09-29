REFERENCE_DEVICE = 'target'

import torch
def run(k, k_nope, k_rope):
    num_heads = k.shape[1]
    rope = k_rope.expand(-1, num_heads, -1)
    return torch.cat([k_nope, rope], dim=-1).to(k.dtype)
