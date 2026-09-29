REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def run(scores, num_expert_group, topk_group, topk, renormalize, routed_scaling_factor, bias, scoring_func):
    return _custom_ops.grouped_topk(scores, num_expert_group, topk_group, topk, renormalize, routed_scaling_factor, bias, scoring_func)
