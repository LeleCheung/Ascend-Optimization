REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def run(input, num_tokens, topk):
    hidden_size = input.shape[1]
    output = torch.zeros(num_tokens, hidden_size, device=input.device, dtype=input.dtype)
    _custom_ops.moe_sum(input, output)
    return output
