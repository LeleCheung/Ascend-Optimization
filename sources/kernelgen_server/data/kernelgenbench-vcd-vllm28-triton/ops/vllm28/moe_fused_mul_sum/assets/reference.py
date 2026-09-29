"""vLLM 参考实现：moe_fused_mul_sum。MoE 专家输出加权求和（Triton）。"""
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum as _vllm_moe_fused_mul_sum,
)


def _baseline_moe_fused_mul_sum(
    inputs, topk_weights, outputs=None, topk_ids=None, expert_map=None
):
    return _vllm_moe_fused_mul_sum(inputs, topk_weights, outputs, topk_ids, expert_map)


def moe_fused_mul_sum(*args, **kwargs):
    return _baseline_moe_fused_mul_sum(*args, **kwargs)
