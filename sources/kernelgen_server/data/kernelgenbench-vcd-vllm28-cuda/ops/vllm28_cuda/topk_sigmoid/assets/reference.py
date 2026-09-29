"""vLLM 参考实现：topk_sigmoid。Sigmoid 打分的 MoE top-k 路由（in-place 写 topk_weights/ids/token_expert_indices）。CUDA 算子。"""
from vllm._custom_ops import topk_sigmoid as _vllm_topk_sigmoid


def _baseline_topk_sigmoid(
    topk_weights,
    topk_ids,
    token_expert_indices,
    gating_output,
    renormalize=False,
    e_score_correction_bias=None,
    routed_scaling_factor=1.0,
    is_padding=None,
):
    _vllm_topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
        is_padding,
    )


def topk_sigmoid(*args, **kwargs):
    return _baseline_topk_sigmoid(*args, **kwargs)
