"""vLLM 参考实现：triton_w4a16_gemm。W4A16 GPTQ-packed int4 GEMM（Triton）。"""
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm as _vllm_triton_w4a16_gemm,
)


def _baseline_triton_w4a16_gemm(a, b_q, scales, qzeros, group_size, zp_bias=8):
    return _vllm_triton_w4a16_gemm(a, b_q, scales, qzeros, group_size, zp_bias)


def triton_w4a16_gemm(*args, **kwargs):
    return _baseline_triton_w4a16_gemm(*args, **kwargs)
