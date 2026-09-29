"""vLLM 参考实现：fused_olmo_hybrid_gdn_gating。Olmo 混合 GDN g/beta 融合门控（Triton）。"""
from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
    fused_olmo_hybrid_gdn_gating as _vllm_fused_olmo_hybrid_gdn_gating,
)


def _baseline_fused_olmo_hybrid_gdn_gating(
    A_log, a, b, dt_bias, allow_neg_eigval=False, beta=1.0, threshold=20.0
):
    return _vllm_fused_olmo_hybrid_gdn_gating(
        A_log, a, b, dt_bias, allow_neg_eigval, beta, threshold
    )


def fused_olmo_hybrid_gdn_gating(*args, **kwargs):
    return _baseline_fused_olmo_hybrid_gdn_gating(*args, **kwargs)
