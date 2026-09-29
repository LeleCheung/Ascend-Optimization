"""vLLM 参考实现：minimax_m3_index_topk。从预计算 score 中选出稀疏 index top-k。"""
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_topk as _vllm_minimax_m3_index_topk,
)


def _baseline_minimax_m3_index_topk(
    score,
    cu_seqlens_q,
    prefix_lens,
    max_query_len,
    topk,
    init_blocks,
    local_blocks,
    out=None,
):
    return _vllm_minimax_m3_index_topk(
        score,
        cu_seqlens_q,
        prefix_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
        out,
    )


def minimax_m3_index_topk(*args, **kwargs):
    return _baseline_minimax_m3_index_topk(*args, **kwargs)
