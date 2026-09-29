REFERENCE_DEVICE = 'target'

import torch
def run(kv_score, head_dim):
    x = kv_score.to(torch.float32)
    kv, score = x[..., :head_dim], x[..., head_dim:]
    w = torch.softmax(score, dim=1)
    return (kv * w).sum(dim=1).to(kv_score.dtype)
