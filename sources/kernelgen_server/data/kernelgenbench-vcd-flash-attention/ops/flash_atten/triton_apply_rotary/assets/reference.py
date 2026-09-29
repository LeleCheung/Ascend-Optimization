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
    from flash_attn.ops.triton import rotary as _m
except (ModuleNotFoundError, ImportError):
    _m = None


def triton_apply_rotary(
    x,
    cos,
    sin,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
):
    """FlashAttention Triton baseline for apply_rotary.

    Applies rotary position embeddings (RoPE) to the first rotary_dim channels
    of x, leaving the rest unchanged.

    Args:
        x (Tensor): input, shape (batch, seqlen, nheads, headdim) when
            cu_seqlens is None, else (total_seqlen, nheads, headdim).
        cos (Tensor): cosine table, shape (seqlen_ro, rotary_dim / 2).
        sin (Tensor): sine table, shape (seqlen_ro, rotary_dim / 2).
        seqlen_offsets (int|Tensor): per-batch position offset(s).
        cu_seqlens (Tensor|None): cumulative seqlens for varlen input.
        max_seqlen (int|None): max seqlen (required when cu_seqlens is set).
        interleaved (bool): GPT-J-style interleaved rotary layout.
        inplace (bool): write the result into x in-place.
        conjugate (bool): use the conjugate rotation (for backward).

    Returns:
        Tensor: rotated output, same shape as x.
    """
    return _m.apply_rotary(
        x, cos, sin,
        seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        interleaved=interleaved, inplace=inplace, conjugate=conjugate,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: triton_apply_rotary requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: triton_apply_rotary requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    def ref_rotary_noninterleaved(x, cos, sin):
        # x: (b, s, h, d); rotate first rotary_dim = 2 * cos.shape[-1] channels.
        rd = 2 * cos.shape[-1]
        xf = x.float()
        x_rot, x_pass = xf[..., :rd], xf[..., rd:]
        x1, x2 = x_rot[..., : rd // 2], x_rot[..., rd // 2:]
        # broadcast cos/sin (s, rd/2) -> (1, s, 1, rd/2)
        c = cos.float()[None, :, None, :]
        s = sin.float()[None, :, None, :]
        o1 = x1 * c - x2 * s
        o2 = x1 * s + x2 * c
        return torch.cat([o1, o2, x_pass], dim=-1)

    for dtype in (torch.float16, torch.bfloat16):
        b, sq, h, d, rd = 2, 128, 8, 64, 64
        x = torch.randn(b, sq, h, d, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rd, 2, device=device, dtype=torch.float32) / rd))
        t = torch.arange(sq, device=device, dtype=torch.float32).unsqueeze(1)
        freqs = t * inv_freq
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)

        out = triton_apply_rotary(x, cos, sin, interleaved=False)
        ref = ref_rotary_noninterleaved(x, cos, sin)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
        print(f"pass: triton_apply_rotary dtype={dtype}")

    print("\ntriton_apply_rotary all tests passed")
