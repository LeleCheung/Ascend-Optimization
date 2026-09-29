"""vLLM 参考实现：awq_dequantize_triton。AWQ 4bit 权重反量化（Triton）。"""
from vllm.model_executor.layers.quantization.awq_triton import (
    awq_dequantize_triton as _vllm_awq_dequantize_triton,
)


def _baseline_awq_dequantize_triton(qweight, scales, zeros, block_size_x=32, block_size_y=32):
    return _vllm_awq_dequantize_triton(qweight, scales, zeros, block_size_x, block_size_y)


def awq_dequantize_triton(*args, **kwargs):
    return _baseline_awq_dequantize_triton(*args, **kwargs)
