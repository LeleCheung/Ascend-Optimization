"""vLLM 参考实现：context_attention_fwd。前缀 prefill 注意力（Triton）。"""
from vllm.v1.attention.ops.prefix_prefill import (
    context_attention_fwd as _vllm_context_attention_fwd,
)


def _baseline_context_attention_fwd(
    q,
    k,
    v,
    o,
    kv_cache_dtype,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_seq_len,
    max_input_len,
    k_scale,
    v_scale,
):
    _vllm_context_attention_fwd(
        q,
        k,
        v,
        o,
        kv_cache_dtype,
        k_cache,
        v_cache,
        b_loc,
        b_start_loc,
        b_seq_len,
        max_seq_len,
        max_input_len,
        k_scale,
        v_scale,
    )


def context_attention_fwd(*args, **kwargs):
    return _baseline_context_attention_fwd(*args, **kwargs)
