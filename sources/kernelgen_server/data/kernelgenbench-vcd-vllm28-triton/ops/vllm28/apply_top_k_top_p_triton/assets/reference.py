"""vLLM 参考实现：apply_top_k_top_p_triton。用 Triton 对 logits 施加 top-k/top-p 掩码。"""
from vllm.v1.sample.ops.topk_topp_triton import (
    apply_top_k_top_p_triton as _vllm_apply_top_k_top_p_triton,
)


def _baseline_apply_top_k_top_p_triton(logits, k, p, mask_value=float("-inf")):
    return _vllm_apply_top_k_top_p_triton(logits, k, p, mask_value)


def apply_top_k_top_p_triton(*args, **kwargs):
    return _baseline_apply_top_k_top_p_triton(*args, **kwargs)
