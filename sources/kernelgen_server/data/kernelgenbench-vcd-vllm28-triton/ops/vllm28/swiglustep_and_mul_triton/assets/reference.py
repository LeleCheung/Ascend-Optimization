"""vLLM 参考实现：swiglustep_and_mul_triton。SwiGLU-step 门控乘法（Triton）。in-place 写 output。"""
from vllm.model_executor.layers.activation import (
    swiglustep_and_mul_triton as _vllm_swiglustep_and_mul_triton,
)


def _baseline_swiglustep_and_mul_triton(output, input, limit=7.0):
    return _vllm_swiglustep_and_mul_triton(output, input, limit)


def swiglustep_and_mul_triton(*args, **kwargs):
    return _baseline_swiglustep_and_mul_triton(*args, **kwargs)
