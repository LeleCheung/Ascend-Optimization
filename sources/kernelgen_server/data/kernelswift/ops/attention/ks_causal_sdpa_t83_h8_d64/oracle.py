REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F

def run(query, key, value):
    num_heads = 8
    head_size = 64
    scale = 1.0 / (head_size ** 0.5)
    num_tokens = query.shape[0]
    q = query.unsqueeze(0).transpose(1, 2)
    k = key.unsqueeze(0).transpose(1, 2)
    v = value.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=True
    )
    return out.squeeze(0).transpose(0, 1).reshape(
        num_tokens, num_heads * head_size
    )
