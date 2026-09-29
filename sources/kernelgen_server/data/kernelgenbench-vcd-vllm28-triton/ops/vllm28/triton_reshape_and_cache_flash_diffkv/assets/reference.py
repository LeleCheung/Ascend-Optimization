"""vLLM 参考实现：triton_reshape_and_cache_flash_diffkv。写入 K/V head_size 不同的分页 cache（Triton）。"""
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv as _vllm_triton_reshape_and_cache_flash_diffkv,
)


def _baseline_triton_reshape_and_cache_flash_diffkv(
    key,
    value,
    kv_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    _vllm_triton_reshape_and_cache_flash_diffkv(
        key,
        value,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def triton_reshape_and_cache_flash_diffkv(*args, **kwargs):
    return _baseline_triton_reshape_and_cache_flash_diffkv(*args, **kwargs)
