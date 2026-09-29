"""vLLM 参考实现：rms_norm_gated。带门控 (z) 的 RMSNorm（Triton）。"""
from vllm.model_executor.layers.mamba.ops.layernorm_gated import (
    rms_norm_gated as _vllm_rms_norm_gated,
)


def _baseline_rms_norm_gated(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True
):
    return _vllm_rms_norm_gated(
        x, weight, bias, z, eps, group_size, norm_before_gate
    )


def rms_norm_gated(*args, **kwargs):
    return _baseline_rms_norm_gated(*args, **kwargs)
