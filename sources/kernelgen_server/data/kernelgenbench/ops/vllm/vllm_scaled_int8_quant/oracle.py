REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def run(input, scale, azp, symmetric):
    return _custom_ops.scaled_int8_quant(input, scale, azp, symmetric)
