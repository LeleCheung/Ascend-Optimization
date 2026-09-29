"""vLLM 参考实现：silu_mul_quant_fp8_packed_triton。SiLU-mul + packed FP8 量化（Triton）。"""
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton as _vllm_silu_mul_quant_fp8_packed_triton,
)


def _baseline_silu_mul_quant_fp8_packed_triton(
    input, group_size=128, output_q=None, clamp_limit=None, alpha=1.0, beta=0.0
):
    return _vllm_silu_mul_quant_fp8_packed_triton(
        input, group_size, output_q, clamp_limit, alpha, beta
    )


def silu_mul_quant_fp8_packed_triton(*args, **kwargs):
    return _baseline_silu_mul_quant_fp8_packed_triton(*args, **kwargs)
