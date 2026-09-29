"""vLLM 参考实现：get_num_nans。统计每行 logits 中的 NaN 数量。"""
from vllm.v1.worker.gpu.metrics.logits import get_num_nans as _vllm_get_num_nans


def _baseline_get_num_nans(logits):
    return _vllm_get_num_nans(logits)


def get_num_nans(*args, **kwargs):
    return _baseline_get_num_nans(*args, **kwargs)
