REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F

def run(query, key, value):
    num_heads = 8
    head_size = 64
    num_kv_heads = 8
    scale = 1.0 / (head_size ** 0.5)
    batch_size, query_length = query.size()[:2]
    key_length = key.size(1)
    q = query.view(
        batch_size, query_length, num_heads, head_size
    ).transpose(1, 2)
    k = key.view(
        batch_size, key_length, num_kv_heads, head_size
    ).transpose(1, 2)
    v = value.view(
        batch_size, key_length, num_kv_heads, head_size
    ).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    return out.transpose(1, 2).reshape(
        batch_size, query_length, -1
    )
