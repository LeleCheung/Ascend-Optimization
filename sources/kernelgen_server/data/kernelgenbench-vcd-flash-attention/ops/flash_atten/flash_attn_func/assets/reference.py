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


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """FlashAttention-2 baseline for flash_attn_func (csrc/flash_attn).

    Standard scaled-dot-product attention with separate q/k/v tensors.
    Supports MQA/GQA when k/v have fewer heads than q.

    Args:
        q (Tensor): queries, shape (batch, seqlen_q, nheads, headdim).
        k (Tensor): keys,   shape (batch, seqlen_k, nheads_k, headdim).
        v (Tensor): values, shape (batch, seqlen_k, nheads_k, headdim).
        dropout_p (float): attention dropout (0.0 during eval).
        softmax_scale (float|None): softmax scale, defaults to 1/sqrt(headdim).
        causal (bool): apply causal mask.
        window_size (tuple): local attention window (-1,-1 = full context).
        softcap (float): logit soft-capping (0.0 disables).
        alibi_slopes (Tensor|None): ALiBi slopes.
        deterministic (bool): use deterministic backward.
        return_attn_probs (bool): also return softmax probs (testing only).

    Returns:
        Tensor: attention output, shape (batch, seqlen_q, nheads, headdim).
    """
    return _m.flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal, window_size,
        softcap, alibi_slopes, deterministic, return_attn_probs,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: flash_attn_func requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: flash_attn_func requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    def ref_attn(q, k, v, causal):
        # (b, s, h, d) -> (b, h, s, d) for torch SDPA
        qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        return o.transpose(1, 2)

    for dtype in (torch.float16, torch.bfloat16):
        for causal in (False, True):
            b, s, h, d = 2, 256, 8, 64
            q = torch.randn(b, s, h, d, device=device, dtype=dtype)
            k = torch.randn(b, s, h, d, device=device, dtype=dtype)
            v = torch.randn(b, s, h, d, device=device, dtype=dtype)

            out = flash_attn_func(q, k, v, causal=causal)
            ref = ref_attn(q, k, v, causal)
            torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
            print(f"pass: flash_attn_func dtype={dtype} causal={causal}")

    print("\nflash_attn_func all tests passed")
