"""vLLM 参考实现：splitk_reduce_triton。split-K 部分和沿第 0 维归约到 out（Triton, in-place）。"""
import torch
try:
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (
        splitk_reduce_triton as _vllm_splitk_reduce_triton,
    )
except ModuleNotFoundError:
    _vllm_splitk_reduce_triton = None


def _baseline_splitk_reduce_triton(partials, out):
    if _vllm_splitk_reduce_triton is None:
        raise RuntimeError("vLLM not installed or module not found")
    return _vllm_splitk_reduce_triton(partials, out)


def splitk_reduce_triton(*args, **kwargs):
    return _baseline_splitk_reduce_triton(*args, **kwargs)
