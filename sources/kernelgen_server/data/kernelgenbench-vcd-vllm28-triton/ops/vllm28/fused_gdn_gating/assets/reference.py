"""vLLM 参考实现：fused_gdn_gating。Gated Delta Net 的 g / beta 融合门控（Triton）。"""
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fused_gdn_gating as _vllm_fused_gdn_gating,
)


def _baseline_fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    return _vllm_fused_gdn_gating(A_log, a, b, dt_bias, beta, threshold)


def fused_gdn_gating(*args, **kwargs):
    return _baseline_fused_gdn_gating(*args, **kwargs)
