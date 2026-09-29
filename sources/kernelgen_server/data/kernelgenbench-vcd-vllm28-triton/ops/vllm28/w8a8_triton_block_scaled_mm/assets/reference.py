"""vLLM 参考实现：w8a8_triton_block_scaled_mm。W8A8 block-wise scaled 矩阵乘（Triton）。"""
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm as _vllm_w8a8_triton_block_scaled_mm,
)


def _baseline_w8a8_triton_block_scaled_mm(A, B, As, Bs, block_size, output_dtype=None):
    import torch
    if output_dtype is None:
        output_dtype = torch.float16
    return _vllm_w8a8_triton_block_scaled_mm(A, B, As, Bs, block_size, output_dtype)


def w8a8_triton_block_scaled_mm(*args, **kwargs):
    return _baseline_w8a8_triton_block_scaled_mm(*args, **kwargs)
