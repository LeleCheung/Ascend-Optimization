REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def run(gating_output, top_k, renormalize):
    num_tokens = gating_output.shape[0]
    device = gating_output.device
    topk_weights = torch.empty(num_tokens, top_k, device=device, dtype=torch.float32)
    topk_ids = torch.empty(num_tokens, top_k, device=device, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, top_k, device=device, dtype=torch.int32)
    _custom_ops.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize)
    return topk_weights, topk_ids
