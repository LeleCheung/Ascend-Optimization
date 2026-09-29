"""vLLM 参考实现：swiglu_limit_func。MoE pad-aware SwiGLU（gate*up，可 clamp）（Triton）。in-place。"""
from vllm.model_executor.layers.fused_moe.utils import (
    swiglu_limit_func as _vllm_swiglu_limit_func,
)


def _baseline_swiglu_limit_func(
    output, input, swiglu_limit=0.0, topk_ids=None, expert_map=None
):
    return _vllm_swiglu_limit_func(output, input, swiglu_limit, topk_ids, expert_map)


def swiglu_limit_func(*args, **kwargs):
    return _baseline_swiglu_limit_func(*args, **kwargs)
