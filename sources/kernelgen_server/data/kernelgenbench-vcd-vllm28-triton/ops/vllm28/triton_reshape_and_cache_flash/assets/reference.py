"""vLLM 参考实现：triton_reshape_and_cache_flash。将 key/value 写入分页 KV cache（Triton）。"""
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash as _vllm_triton_reshape_and_cache_flash,
)


def _baseline_triton_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    _vllm_triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def triton_reshape_and_cache_flash(*args, **kwargs):
    return _baseline_triton_reshape_and_cache_flash(*args, **kwargs)
