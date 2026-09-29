"""vLLM 参考实现：topk_hash_softplus_sqrt。Softplus/sqrt 打分的 MoE top-k 路由（in-place 写出）。CUDA 算子。"""
from vllm._custom_ops import topk_hash_softplus_sqrt as _vllm_topk_hash_softplus_sqrt


def _baseline_topk_hash_softplus_sqrt(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize=False,
    routed_scaling_factor=1.0,
    e_score_correction_bias=None,
    input_tokens=None,
    hash_indices_table=None,
    is_padding=None,
):
    _vllm_topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
        is_padding,
    )


def topk_hash_softplus_sqrt(*args, **kwargs):
    return _baseline_topk_hash_softplus_sqrt(*args, **kwargs)
