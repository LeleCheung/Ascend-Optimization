"""vLLM 参考实现：dsv3_fused_a_gemm。低延迟 fused-A GEMM (SM90+, bf16, 1-16 tokens)。CUDA 算子，in-place 写 output。"""
from vllm._custom_ops import dsv3_fused_a_gemm as _vllm_dsv3_fused_a_gemm


def _baseline_dsv3_fused_a_gemm(output, mat_a, mat_b, enable_pdl=False):
    _vllm_dsv3_fused_a_gemm(output, mat_a, mat_b, enable_pdl)
    return output


def dsv3_fused_a_gemm(*args, **kwargs):
    return _baseline_dsv3_fused_a_gemm(*args, **kwargs)
