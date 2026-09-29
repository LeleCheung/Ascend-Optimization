REFERENCE_DEVICE = 'target'

import torch
def run(topk_ids, topk_weights):
    weight_bits = (
        topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    )
    return (topk_ids.to(torch.int32) << 16) | weight_bits
