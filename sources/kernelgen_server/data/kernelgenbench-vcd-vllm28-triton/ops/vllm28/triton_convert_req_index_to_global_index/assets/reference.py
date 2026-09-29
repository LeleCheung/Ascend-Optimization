"""vLLM 参考实现：triton_convert_req_index_to_global_index。把每请求局部索引转成全局 KV cache 槽位（Triton）。"""
import torch
try:
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_convert_req_index_to_global_index as _vllm_triton_convert_req_index_to_global_index,
    )
except ModuleNotFoundError:
    _vllm_triton_convert_req_index_to_global_index = None


def _baseline_triton_convert_req_index_to_global_index(
    req_id, block_table, token_indices, **kwargs
):
    if _vllm_triton_convert_req_index_to_global_index is None:
        raise RuntimeError("vLLM not installed or module not found")
    return _vllm_triton_convert_req_index_to_global_index(
        req_id, block_table, token_indices, **kwargs
    )


def triton_convert_req_index_to_global_index(*args, **kwargs):
    return _baseline_triton_convert_req_index_to_global_index(*args, **kwargs)
