"""vLLM 参考实现：rmsnorm_nw。无权重 RMSNorm（Triton）。"""
from vllm.model_executor.kernels.mhc.triton import rmsnorm_nw as _vllm_rmsnorm_nw


def _baseline_rmsnorm_nw(x, eps):
    return _vllm_rmsnorm_nw(x, eps)


def rmsnorm_nw(*args, **kwargs):
    return _baseline_rmsnorm_nw(*args, **kwargs)
