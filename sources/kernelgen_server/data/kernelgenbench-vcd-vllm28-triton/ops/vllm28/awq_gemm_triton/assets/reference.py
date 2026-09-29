"""vLLM 参考实现：awq_gemm_triton。AWQ 4bit 量化 GEMM（Triton）。"""
from vllm.model_executor.layers.quantization.awq_triton import (
    awq_gemm_triton as _vllm_awq_gemm_triton,
)


def _baseline_awq_gemm_triton(
    input, qweight, scales, qzeros, split_k_iters,
    block_size_m=32, block_size_n=32, block_size_k=32,
):
    return _vllm_awq_gemm_triton(
        input, qweight, scales, qzeros, split_k_iters,
        block_size_m, block_size_n, block_size_k,
    )


def awq_gemm_triton(*args, **kwargs):
    return _baseline_awq_gemm_triton(*args, **kwargs)
