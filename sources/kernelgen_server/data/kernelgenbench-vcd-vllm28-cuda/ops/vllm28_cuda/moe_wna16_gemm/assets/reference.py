"""vLLM 参考实现：moe_wna16_gemm。WNA16 (Weight-only INT4 Activation FP16) MoE GEMM，in-place 写 output。CUDA 算子。"""
from vllm._custom_ops import moe_wna16_gemm as _vllm_moe_wna16_gemm


def _baseline_moe_wna16_gemm(
    input,
    output,
    b_qweight,
    b_scales,
    b_qzeros,
    topk_weights,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    top_k,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    bit,
):
    _vllm_moe_wna16_gemm(
        input,
        output,
        b_qweight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        top_k,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        bit,
    )


def moe_wna16_gemm(*args, **kwargs):
    return _baseline_moe_wna16_gemm(*args, **kwargs)
