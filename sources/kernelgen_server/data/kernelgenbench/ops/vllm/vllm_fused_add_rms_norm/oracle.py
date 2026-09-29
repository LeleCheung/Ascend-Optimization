REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def run(input, residual, weight, epsilon):
    _custom_ops.fused_add_rms_norm(input, residual, weight, epsilon)
    return input
