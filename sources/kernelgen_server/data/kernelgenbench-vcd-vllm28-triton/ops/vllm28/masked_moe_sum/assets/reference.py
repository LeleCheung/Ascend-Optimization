"""vLLM 参考实现：masked_moe_sum。按 topk_ids 掩码对 MoE intermediate 求和（Triton）。in-place。"""
from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    masked_moe_sum as _vllm_masked_moe_sum,
)


def _baseline_masked_moe_sum(intermediate, topk_ids, output):
    return _vllm_masked_moe_sum(intermediate, topk_ids, output)


def masked_moe_sum(*args, **kwargs):
    return _baseline_masked_moe_sum(*args, **kwargs)
