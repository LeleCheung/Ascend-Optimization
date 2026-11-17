# Copyright 2026, The FlagOS Contributors.
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
"""Triton implementation of ``aten.upsample_bilinear2d`` (NCHW, bilinear).

Design notes (round 1)
-----------------------
* One flat 1D tile of output elements per program.  The output is a freshly
  allocated contiguous ``[N, C, OH, OW]`` tensor, therefore the flat element
  index ``offs`` *is* the output offset: it removes every modulo from the store
  path and turns the write into one contiguous vector store (metadata zero
  copy view is not even needed).
* ``N, C, IH, IW, OH, OW`` and ``ALIGN_CORNERS`` are ``constexpr`` so all
  integer div/mod become constant folds (shifts for the power-of-two timing
  shapes) and the ``OH == 1 / OW == 1`` degenerate cases are resolved with
  Python-level ``if`` -- never with a ``tl.where`` that would keep a division
  by ``(OH - 1)`` alive in the IR.  That division-by-zero in the two-branch
  ``tl.where`` form is what broke ``ConvertTritonIRToLinalgIR`` for the 1x1
  outputs in the previous candidate.
* Input addressing uses the *real* strides, so non-contiguous inputs
  (``stride(2)`` slices, transposes) read the true storage.  The previous
  candidate hard-coded contiguous NCHW and produced large numerical errors.
* Coordinate mapping follows PyTorch exactly:
  ``align_corners`` -> ``src = idx * (in-1)/(out-1)`` (0 when out == 1),
  otherwise ``src = max((idx + 0.5) * scale - 0.5, 0)`` with
  ``scale = 1/scales_*`` when provided and positive, else ``in/out``.
  The right/bottom neighbour is ``min(i0 + 1, size - 1)`` which reproduces
  PyTorch's ``i1p = (i1 < size - 1) ? 1 : 0`` edge behaviour bit for bit in
  the interpolation expression.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _upsample_bilinear2d_kernel(
    ptr_o,
    ptr_i,
    total,
    s_n,
    s_c,
    s_h,
    s_w,
    rh,
    rw,
    C: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    row = offs // OW
    oh = row % OH
    nc = row // OH

    if ALIGN_CORNERS:
        if OH > 1:
            real_h = oh.to(tl.float32) * ((IH - 1.0) / (OH - 1.0))
        else:
            real_h = tl.zeros([BLOCK], dtype=tl.float32)
        if OW > 1:
            real_w = ow.to(tl.float32) * ((IW - 1.0) / (OW - 1.0))
        else:
            real_w = tl.zeros([BLOCK], dtype=tl.float32)
    else:
        real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
        real_w = tl.maximum((ow.to(tl.float32) + 0.5) * rw - 0.5, 0.0)

    h0 = real_h.to(tl.int32)
    w0 = real_w.to(tl.int32)
    hl = real_h - h0.to(tl.float32)
    wl = real_w - w0.to(tl.float32)

    h0 = tl.minimum(h0, IH - 1)
    w0 = tl.minimum(w0, IW - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    w1 = tl.minimum(w0 + 1, IW - 1)

    n = nc // C
    c = nc - n * C
    base = n * s_n + c * s_c
    r0 = base + h0 * s_h
    r1 = base + h1 * s_h
    c0 = w0 * s_w
    c1 = w1 * s_w

    v00 = tl.load(ptr_i + (r0 + c0), mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(ptr_i + (r0 + c1), mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(ptr_i + (r1 + c0), mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(ptr_i + (r1 + c1), mask=mask, other=0.0).to(tl.float32)

    # PyTorch's association order: h0l*(w0l*v00 + w1l*v01) + h1l*(w0l*v10 + w1l*v11)
    w0l = 1.0 - wl
    h0l = 1.0 - hl
    top = w0l * v00 + wl * v01
    bot = w0l * v10 + wl * v11
    val = h0l * top + hl * bot

    tl.store(ptr_o + offs, val.to(ptr_o.dtype.element_ty), mask=mask)


def _reciprocal_scale(in_size, out_size, align_corners, scale):
    """PyTorch ``area_pixel_compute_scale`` (accscalar_t == float32 path)."""
    if align_corners:
        if out_size > 1:
            return (in_size - 1) / (out_size - 1)
        return 0.0
    if scale is not None and scale > 0:
        return 1.0 / float(scale)
    return in_size / out_size


def _pick_block(total):
    if total >= (1 << 20):
        return 2048
    if total >= (1 << 16):
        return 1024
    if total >= (1 << 12):
        return 512
    return 256


def run(input, output_size, align_corners=False, scales_h=None, scales_w=None):
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    N, C, IH, IW = input.shape
    OH = int(output_size[0])
    OW = int(output_size[1])

    out = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    total = N * C * OH * OW
    if total == 0:
        return out

    rh = _reciprocal_scale(IH, OH, bool(align_corners), scales_h)
    rw = _reciprocal_scale(IW, OW, bool(align_corners), scales_w)

    s_n, s_c, s_h, s_w = input.stride()
    block = _pick_block(total)
    grid = (triton.cdiv(total, block),)

    _upsample_bilinear2d_kernel[grid](
        out,
        input,
        total,
        s_n,
        s_c,
        s_h,
        s_w,
        rh,
        rw,
        C=C,
        OH=OH,
        OW=OW,
        IH=IH,
        IW=IW,
        ALIGN_CORNERS=bool(align_corners),
        BLOCK=block,
    )
    return out
