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

try:
    from flash_attn import flash_attn_interface as _m
except (ModuleNotFoundError, ImportError):
    _m = None


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    """FlashAttention-2 baseline for flash_attn_with_kvcache (csrc/flash_attn).

    Attention against a KV cache, for incremental decoding. If k/v are given,
    k_cache/v_cache are updated in-place with the new keys/values before the
    attention is computed.

    Args:
        q (Tensor): queries, shape (batch, seqlen_q, nheads, headdim).
        k_cache (Tensor): key cache, shape (batch, seqlen_cache, nheads_k, headdim)
            (or paged layout when block_table is given).
        v_cache (Tensor): value cache, same layout as k_cache.
        k (Tensor|None): new keys to append to the cache.
        v (Tensor|None): new values to append to the cache.
        rotary_cos/rotary_sin (Tensor|None): optional RoPE tables applied to q/k.
        cache_seqlens (int|Tensor|None): current cache length per batch.
        cache_batch_idx (Tensor|None): indices into the cache batch dim.
        cache_leftpad (Tensor|None): left padding per batch.
        block_table (Tensor|None): paged-KV block table.
        softmax_scale (float|None): softmax scale, defaults to 1/sqrt(headdim).
        causal (bool): apply causal mask.
        window_size (tuple): local attention window (-1,-1 = full context).
        softcap (float): logit soft-capping (0.0 disables).
        rotary_interleaved (bool): interleaved RoPE layout.
        alibi_slopes (Tensor|None): ALiBi slopes.
        num_splits (int): split-KV parallelism (0 = auto).
        return_softmax_lse (bool): also return the softmax LSE.

    Returns:
        Tensor: attention output, shape (batch, seqlen_q, nheads, headdim)
        (plus LSE when return_softmax_lse=True).
    """
    return _m.flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v,
        rotary_cos=rotary_cos, rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens, cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad, block_table=block_table,
        softmax_scale=softmax_scale, causal=causal, window_size=window_size,
        softcap=softcap, rotary_interleaved=rotary_interleaved,
        alibi_slopes=alibi_slopes, num_splits=num_splits,
        return_softmax_lse=return_softmax_lse,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: flash_attn_with_kvcache requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: flash_attn_with_kvcache requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    def ref_attn(q, k, v, causal):
        qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        return o.transpose(1, 2)

    for dtype in (torch.float16, torch.bfloat16):
        # Decode: seqlen_q=1 attending over the full cache (no causal mask needed).
        b, sq, sc, h, d = 2, 1, 256, 8, 64
        q = torch.randn(b, sq, h, d, device=device, dtype=dtype)
        k_cache = torch.randn(b, sc, h, d, device=device, dtype=dtype)
        v_cache = torch.randn(b, sc, h, d, device=device, dtype=dtype)

        out = flash_attn_with_kvcache(q, k_cache, v_cache)
        ref = ref_attn(q, k_cache, v_cache, causal=False)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
        print(f"pass: flash_attn_with_kvcache dtype={dtype}")

    print("\nflash_attn_with_kvcache all tests passed")
