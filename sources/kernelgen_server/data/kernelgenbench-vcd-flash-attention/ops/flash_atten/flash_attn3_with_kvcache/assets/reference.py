# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

_m = None
try:
    from flash_attn.hopper import flash_attn_interface as _m
except (ModuleNotFoundError, ImportError):
    try:
        import flash_attn_interface as _m  # standalone flash-attn-3 wheel
    except (ModuleNotFoundError, ImportError):
        _m = None


def flash_attn3_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    num_splits=0,
    return_softmax_lse=False,
):
    """FlashAttention-3 baseline for flash_attn_with_kvcache (Hopper / SM90 only).

    Attention against a KV cache for incremental decoding, using the FA3
    Hopper kernels. If k/v are given, the caches are updated in-place first.

    Args:
        q (Tensor): queries, shape (batch, seqlen_q, nheads, headdim).
        k_cache (Tensor): key cache, shape (batch, seqlen_cache, nheads_k, headdim)
            (or paged layout when a page table is used).
        v_cache (Tensor): value cache, same layout as k_cache.
        k (Tensor|None): new keys to append to the cache.
        v (Tensor|None): new values to append to the cache.
        cache_seqlens (int|Tensor|None): current cache length per batch.
        softmax_scale (float|None): softmax scale, defaults to 1/sqrt(headdim).
        causal (bool): apply causal mask.
        window_size (tuple): local attention window (-1,-1 = full context).
        softcap (float): logit soft-capping (0.0 disables).
        num_splits (int): split-KV parallelism (0 = auto).
        return_softmax_lse (bool): also return the softmax LSE.

    Returns:
        Tensor out, or (out, softmax_lse) when return_softmax_lse=True.
        out has shape (batch, seqlen_q, nheads, headdim).
    """
    return _m.flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v,
        cache_seqlens=cache_seqlens, softmax_scale=softmax_scale,
        causal=causal, window_size=window_size, softcap=softcap,
        num_splits=num_splits, return_softmax_lse=return_softmax_lse,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: flash_attn3_with_kvcache requires flash_attn (Hopper/FA3) to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        print("skip: flash_attn3_with_kvcache requires a Hopper (SM90) GPU")
        raise SystemExit(0)

    device = "cuda"

    def ref_attn(q, k, v, causal):
        qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        return o.transpose(1, 2)

    for dtype in (torch.bfloat16, torch.float16):
        b, sq, sc, h, d = 2, 1, 512, 16, 128
        q = torch.randn(b, sq, h, d, device=device, dtype=dtype)
        k_cache = torch.randn(b, sc, h, d, device=device, dtype=dtype)
        v_cache = torch.randn(b, sc, h, d, device=device, dtype=dtype)
        cache_seqlens = torch.full((b,), sc, dtype=torch.int32, device=device)

        out = flash_attn3_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens)
        out = out[0] if isinstance(out, (tuple, list)) else out
        ref = ref_attn(q, k_cache, v_cache, causal=False)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
        print(f"pass: flash_attn3_with_kvcache dtype={dtype}")

    print("\nflash_attn3_with_kvcache all tests passed")
