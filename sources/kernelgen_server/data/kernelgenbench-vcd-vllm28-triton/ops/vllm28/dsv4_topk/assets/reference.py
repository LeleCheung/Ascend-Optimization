"""vLLM 参考实现：dsv4_topk。DeepSeek-V4 MoE 路由 top-k 选择，返回 (weights, ids)。"""
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import (
    dsv4_topk as _vllm_dsv4_topk,
)


def _baseline_dsv4_topk(gating_output, correction_bias, indices_dtype, routed_scaling_factor):
    return _vllm_dsv4_topk(
        gating_output, correction_bias, indices_dtype, routed_scaling_factor
    )


def dsv4_topk(*args, **kwargs):
    return _baseline_dsv4_topk(*args, **kwargs)
