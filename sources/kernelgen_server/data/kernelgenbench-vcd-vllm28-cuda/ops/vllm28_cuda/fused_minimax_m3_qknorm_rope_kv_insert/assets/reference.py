"""vLLM 参考实现：fused_minimax_m3_qknorm_rope_kv_insert。MiniMax-M3 融合 QKNorm+RoPE+KV cache 插入（in-place）。CUDA 算子。"""
from vllm._custom_ops import (
    fused_minimax_m3_qknorm_rope_kv_insert as _vllm_fused_minimax_m3_qknorm_rope_kv_insert,
)


def _baseline_fused_minimax_m3_qknorm_rope_kv_insert(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    num_heads,
    num_kv_heads,
    rotary_dim,
    eps,
    index_q_norm_weight=None,
    index_k_norm_weight=None,
    num_index_heads=0,
    slot_mapping=None,
    index_slot_mapping=None,
    kv_cache=None,
    index_cache=None,
    block_size=0,
    q_out=None,
    index_q_out=None,
    kv_cache_dtype="auto",
    skip_index_branch=False,
    q_fp8_out=None,
    q_fp8_scale=1.0,
):
    _vllm_fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
        skip_index_branch,
        q_fp8_out,
        q_fp8_scale,
    )
    return qkv


def fused_minimax_m3_qknorm_rope_kv_insert(*args, **kwargs):
    return _baseline_fused_minimax_m3_qknorm_rope_kv_insert(*args, **kwargs)
