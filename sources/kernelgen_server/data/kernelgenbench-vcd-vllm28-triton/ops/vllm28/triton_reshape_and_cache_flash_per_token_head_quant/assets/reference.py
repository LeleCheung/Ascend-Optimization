"""vLLM 参考实现：triton_reshape_and_cache_flash_per_token_head_quant。按 (token, head) 量化写入分页 cache（Triton）。"""
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant as _vllm_op,
)


def _baseline_triton_reshape_and_cache_flash_per_token_head_quant(
    key,
    value,
    key_cache,
    value_cache,
    k_scale_cache,
    v_scale_cache,
    slot_mapping,
    kv_quant_mode,
):
    _vllm_op(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        kv_quant_mode,
    )


def triton_reshape_and_cache_flash_per_token_head_quant(*args, **kwargs):
    return _baseline_triton_reshape_and_cache_flash_per_token_head_quant(*args, **kwargs)
