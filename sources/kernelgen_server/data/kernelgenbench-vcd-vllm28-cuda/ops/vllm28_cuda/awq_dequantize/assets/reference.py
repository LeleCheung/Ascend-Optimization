"""vLLM 参考实现：awq_dequantize。AWQ 4bit 权重反量化为 fp16。CUDA 算子。"""
from vllm._custom_ops import awq_dequantize as _vllm_awq_dequantize


def _baseline_awq_dequantize(qweight, scales, zeros, split_k_iters, thx, thy):
    return _vllm_awq_dequantize(qweight, scales, zeros, split_k_iters, thx, thy)


def awq_dequantize(*args, **kwargs):
    return _baseline_awq_dequantize(*args, **kwargs)
