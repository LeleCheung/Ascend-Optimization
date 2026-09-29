"""vLLM 参考实现：gemma_rmsnorm。Gemma 风格带权 RMSNorm（Triton）。"""
from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
    gemma_rmsnorm as _vllm_gemma_rmsnorm,
)


def _baseline_gemma_rmsnorm(x, weight, eps):
    return _vllm_gemma_rmsnorm(x, weight, eps)


def gemma_rmsnorm(*args, **kwargs):
    return _baseline_gemma_rmsnorm(*args, **kwargs)
