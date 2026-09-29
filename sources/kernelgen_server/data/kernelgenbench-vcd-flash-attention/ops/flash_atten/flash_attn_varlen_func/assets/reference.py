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


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """FlashAttention-2 baseline for flash_attn_varlen_func (csrc/flash_attn).

    Variable-length attention: sequences are concatenated along dim 0 and
    delimited by cumulative-sequence-length tensors. Supports MQA/GQA.

    Args:
        q (Tensor): queries, shape (total_q, nheads, headdim).
        k (Tensor): keys,   shape (total_k, nheads_k, headdim).
        v (Tensor): values, shape (total_k, nheads_k, headdim).
        cu_seqlens_q (Tensor): int32 prefix sums of q seqlens, shape (batch+1,).
        cu_seqlens_k (Tensor): int32 prefix sums of k seqlens, shape (batch+1,).
        max_seqlen_q (int): max query seqlen across the batch.
        max_seqlen_k (int): max key seqlen across the batch.
        dropout_p (float): attention dropout (0.0 during eval).
        softmax_scale (float|None): softmax scale, defaults to 1/sqrt(headdim).
        causal (bool): apply causal mask.
        window_size (tuple): local attention window (-1,-1 = full context).
        softcap (float): logit soft-capping (0.0 disables).
        alibi_slopes (Tensor|None): ALiBi slopes.
        deterministic (bool): use deterministic backward.
        return_attn_probs (bool): also return softmax probs (testing only).
        block_table (Tensor|None): paged-KV block table.

    Returns:
        Tensor: attention output, shape (total_q, nheads, headdim).
    """
    return _m.flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal, window_size, softcap,
        alibi_slopes, deterministic, return_attn_probs, block_table,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: flash_attn_varlen_func requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: flash_attn_varlen_func requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    def ref_varlen(q, k, v, cu, causal):
        # Per-sequence SDPA over the packed (total, h, d) layout.
        outs = []
        for i in range(len(cu) - 1):
            s, e = int(cu[i]), int(cu[i + 1])
            qi = q[s:e].transpose(0, 1).float().unsqueeze(0)  # (1, h, sq, d)
            ki = k[s:e].transpose(0, 1).float().unsqueeze(0)
            vi = v[s:e].transpose(0, 1).float().unsqueeze(0)
            oi = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
            outs.append(oi.squeeze(0).transpose(0, 1))  # (sq, h, d)
        return torch.cat(outs, dim=0)

    for dtype in (torch.float16, torch.bfloat16):
        for causal in (False, True):
            b, s, h, d = 2, 256, 8, 64
            cu = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=device)
            total = b * s
            q = torch.randn(total, h, d, device=device, dtype=dtype)
            k = torch.randn(total, h, d, device=device, dtype=dtype)
            v = torch.randn(total, h, d, device=device, dtype=dtype)

            out = flash_attn_varlen_func(q, k, v, cu, cu, s, s, causal=causal)
            ref = ref_varlen(q, k, v, cu, causal)
            torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
            print(f"pass: flash_attn_varlen_func dtype={dtype} causal={causal}")

    print("\nflash_attn_varlen_func all tests passed")
