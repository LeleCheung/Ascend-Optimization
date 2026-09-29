REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def run(input, weight, epsilon, quant_dtype, group_size, is_scale_transposed):
    quant_dtype = getattr(torch, quant_dtype)
    return _custom_ops.rms_norm_per_block_quant(
        input, weight, epsilon, quant_dtype, group_size, None, None, is_scale_transposed
    )
