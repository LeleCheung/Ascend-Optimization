"""vLLM 参考实现：fast_hadamard_transform。沿最后一维的非归一化 Walsh-Hadamard 变换（Triton）。"""
import torch
try:
    from vllm.v1.attention.ops.int4_per_token_head import (
        fast_hadamard_transform as _vllm_fast_hadamard_transform,
    )
except ModuleNotFoundError:
    _vllm_fast_hadamard_transform = None


def _baseline_fast_hadamard_transform(x):
    if _vllm_fast_hadamard_transform is None:
        raise RuntimeError("vLLM not installed or module not found")
    return _vllm_fast_hadamard_transform(x)


def fast_hadamard_transform(*args, **kwargs):
    return _baseline_fast_hadamard_transform(*args, **kwargs)
