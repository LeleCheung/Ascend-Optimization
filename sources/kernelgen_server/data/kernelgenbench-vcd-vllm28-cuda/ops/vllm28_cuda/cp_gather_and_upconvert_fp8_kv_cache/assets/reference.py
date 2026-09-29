"""vLLM 参考实现：cp_gather_and_upconvert_fp8_kv_cache。从 FP8 KV cache 聚集并上转为 BF16。CUDA 算子（in-place）。"""
from vllm._custom_ops import cp_gather_and_upconvert_fp8_kv_cache as _vllm_cp_gather_and_upconvert_fp8_kv_cache


def _baseline_cp_gather_and_upconvert_fp8_kv_cache(
    src_cache, dst, block_table, workspace_starts, batch_size, seq_starts=None,
):
    # in-place：写入 dst，返回该张量供比较
    _vllm_cp_gather_and_upconvert_fp8_kv_cache(
        src_cache, dst, block_table, workspace_starts, batch_size, seq_starts,
    )
    return dst


def cp_gather_and_upconvert_fp8_kv_cache(*args, **kwargs):
    return _baseline_cp_gather_and_upconvert_fp8_kv_cache(*args, **kwargs)
