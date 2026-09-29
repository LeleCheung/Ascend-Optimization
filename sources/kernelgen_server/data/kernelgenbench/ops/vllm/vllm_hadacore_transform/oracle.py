REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def run(x, inplace):
    x_work = x.clone()
    return _custom_ops.hadacore_transform(x_work, inplace)
