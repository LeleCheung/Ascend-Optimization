"""vLLM 参考实现：gemma4_fused_routing_kernel_triton。Gemma4 MoE 融合路由（Triton）。"""
from vllm.model_executor.models.gemma4 import (
    gemma4_fused_routing_kernel_triton as _vllm_gemma4_fused_routing_kernel_triton,
)


def _baseline_gemma4_fused_routing_kernel_triton(
    gating_output, topk, per_expert_scale, num_warps=1
):
    return _vllm_gemma4_fused_routing_kernel_triton(
        gating_output, topk, per_expert_scale, num_warps
    )


def gemma4_fused_routing_kernel_triton(*args, **kwargs):
    return _baseline_gemma4_fused_routing_kernel_triton(*args, **kwargs)
