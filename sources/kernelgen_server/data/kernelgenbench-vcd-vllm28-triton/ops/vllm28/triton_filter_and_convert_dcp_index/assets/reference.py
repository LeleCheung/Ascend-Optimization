"""vLLM 参考实现：triton_filter_and_convert_dcp_index。过滤出本 DCP rank 拥有的槽位并转全局索引（Triton）。"""
import torch
try:
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index as _vllm_triton_filter_and_convert_dcp_index,
    )
except ModuleNotFoundError:
    _vllm_triton_filter_and_convert_dcp_index = None


def _baseline_triton_filter_and_convert_dcp_index(
    req_id, block_table, token_indices, dcp_size, dcp_rank, **kwargs
):
    if _vllm_triton_filter_and_convert_dcp_index is None:
        raise RuntimeError("vLLM not installed or module not found")
    return _vllm_triton_filter_and_convert_dcp_index(
        req_id, block_table, token_indices, dcp_size, dcp_rank, **kwargs
    )


def triton_filter_and_convert_dcp_index(*args, **kwargs):
    return _baseline_triton_filter_and_convert_dcp_index(*args, **kwargs)
