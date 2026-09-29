"""vLLM 参考实现：fused_norm_rope。DeepSeek-V3.2 融合 RMSNorm + RoPE + cache 写入（Triton）。"""
from vllm.models.deepseek_v32.common.kernels import (
    fused_norm_rope as _vllm_fused_norm_rope,
)


def _baseline_fused_norm_rope(*args, **kwargs):
    return _vllm_fused_norm_rope(*args, **kwargs)


def fused_norm_rope(*args, **kwargs):
    return _baseline_fused_norm_rope(*args, **kwargs)
