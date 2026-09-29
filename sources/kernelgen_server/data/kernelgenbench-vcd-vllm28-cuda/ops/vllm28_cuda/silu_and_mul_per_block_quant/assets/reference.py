"""vLLM 参考实现：silu_and_mul_per_block_quant。融合 SiLU+Mul 后做 per-block 量化，返回 (packed_out, scales)。CUDA 算子。"""
from vllm._custom_ops import silu_and_mul_per_block_quant as _vllm_silu_and_mul_per_block_quant


def _baseline_silu_and_mul_per_block_quant(
    input, group_size, quant_dtype, scale_ub=None, is_scale_transposed=False
):
    return _vllm_silu_and_mul_per_block_quant(
        input, group_size, quant_dtype, scale_ub, is_scale_transposed
    )


def silu_and_mul_per_block_quant(*args, **kwargs):
    return _baseline_silu_and_mul_per_block_quant(*args, **kwargs)
