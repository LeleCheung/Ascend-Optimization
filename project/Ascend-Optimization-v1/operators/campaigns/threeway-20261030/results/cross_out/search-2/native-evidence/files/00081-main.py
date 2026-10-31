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

"""Cross product: cached contiguous launch, full ND stride addressing and out-alias protection."""

import torch
import torch_npu
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# flat kernel (cross axis == last, contiguous)
# --------------------------------------------------------------------------- #
@triton.jit
def _cross_flat_kernel(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    mask = (lane < STEP) & (off < n_elem)
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    phase = lane % 3
    p = tl.minimum(tl.where(phase == 2, lane - 2, lane + 1), STEP - 1)
    q = tl.minimum(tl.where(phase == 0, lane + 2, lane - 1), STEP - 1)
    a1 = tl.gather(a, p, axis=0)
    a2 = tl.gather(a, q, axis=0)
    b1 = tl.gather(b, p, axis=0)
    b2 = tl.gather(b, q, axis=0)
    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)



# --------------------------------------------------------------------------- #
# general kernel (any cross axis, any strides, broadcasting)
# --------------------------------------------------------------------------- #
@triton.jit
def _cross_general_kernel(
    a_ptr, b_ptr, o_ptr, vectors,
    SHAPE: tl.constexpr, AXIS: tl.constexpr,
    SA: tl.constexpr, SB: tl.constexpr, SO: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
    m = index < vectors
    rem = index.to(tl.int64)
    oa = tl.zeros((BLOCK_I,), tl.int64)
    ob = tl.zeros((BLOCK_I,), tl.int64)
    oo = tl.zeros((BLOCK_I,), tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        if axis != AXIS:
            coordinate = rem % SHAPE[axis]
            rem = rem // SHAPE[axis]
            oa += coordinate * SA[axis]
            ob += coordinate * SB[axis]
            oo += coordinate * SO[axis]
    pa, pb = a_ptr + oa, b_ptr + ob
    a0 = tl.load(pa, mask=m, other=0.0)
    a1 = tl.load(pa + SA[AXIS], mask=m, other=0.0)
    a2 = tl.load(pa + 2 * SA[AXIS], mask=m, other=0.0)
    b0 = tl.load(pb, mask=m, other=0.0)
    b1 = tl.load(pb + SB[AXIS], mask=m, other=0.0)
    b2 = tl.load(pb + 2 * SB[AXIS], mask=m, other=0.0)
    po = o_ptr + oo
    tl.store(po, a1 * b2 - a2 * b1, mask=m)
    tl.store(po + SO[AXIS], a2 * b0 - a0 * b2, mask=m)
    tl.store(po + 2 * SO[AXIS], a0 * b1 - a1 * b0, mask=m)


def _pick_block(n_elem, esize):
    # Eight live input tiles (+ one output tile) have to coexist in UB; the
    # compiler double-buffers them, so keep BLOCK small enough for 4-byte
    # elements to fit.
    if n_elem <= 8192:
        return 1024
    if n_elem <= 131072:
        return 1024
    return 1024 if esize == 4 else 2048


def _default_dim(input, other):
    broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
    for axis, size in enumerate(broadcast_shape):
        if size == 3:
            return axis
    raise RuntimeError(
        "no dimension of size 3 in input; cross requires a dimension of length 3"
    )


def _broadcast_strides(t, out_shape):
    """Strides of ``t`` expressed against ``out_shape`` (0 where broadcast)."""
    nd = len(out_shape)
    lead = nd - t.dim()
    res = [0] * nd
    for i in range(t.dim()):
        d = lead + i
        if t.shape[i] == 1 and out_shape[d] != 1:
            res[d] = 0
        else:
            res[d] = t.stride(i)
    return res


_PLANS = {}
_RAW_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)


def _current_stream(device):
    if _RAW_STREAM is not None:
        try:return _RAW_STREAM(device)
        except TypeError:return _RAW_STREAM()
    return torch.npu.current_stream(device).npu_stream


def _launch_flat(a, b, out, n):
    pa, pb, po = a.data_ptr(), b.data_ptr(), out.data_ptr()
    key = (a.device.index, a.dtype, b.dtype, n)
    entry = _PLANS.get(key)
    if entry is None:
        block = _pick_block(n, out.element_size())
        step = (block // 3) * 3
        grid = (n + step - 1) // step
        launch = None
        if not ((pa | pb | po) & 15):
            try:
                compiled = _cross_flat_kernel.warmup(a, b, out, n, BLOCK=block, STEP=step, grid=(1,))
                launcher = compiled.run
                md = launcher.metadata
                if not (getattr(launcher, 'compile_only', False) or
                        getattr(launcher, 'enable_msprof_register_tensor', False) or
                        getattr(md, 'debug_enabled', False) or getattr(md, 'debug', False)):
                    launch = (launcher.launch, compiled.function, compiled.packed_metadata)
            except Exception:
                pass
        entry = (grid, block, step, launch)
        _PLANS[key] = entry
    launch = entry[3]
    if launch and not ((pa | pb | po) & 15):
        launch[0](entry[0], 1, 1, _current_stream(a.device.index), launch[1], launch[2],
                  None, None, None, pa, pb, po, n, entry[1], entry[2])
    else:
        _cross_flat_kernel[(entry[0],)](a, b, out, n, BLOCK=entry[1], STEP=entry[2])


def _overlaps(x, y):
    """Byte-range overlap test between two tensor views (metadata only)."""
    px, py = x.data_ptr(), y.data_ptr()
    nx = x.numel() * x.element_size()
    ny = y.numel() * y.element_size()
    return px < py + ny and py < px + nx


def run(input, other, dim=None, *, out):
    if input.data_ptr() == out.data_ptr() or _overlaps(input, out):
        input = input.clone()
    if other.data_ptr() == out.data_ptr() or _overlaps(other, out):
        other = other.clone()
    if dim is None:
        dim = _default_dim(input, other)
    nd = input.dim()
    if dim < 0:
        dim += nd

    if (
        dim == nd - 1
        and input.shape == out.shape
        and other.shape == out.shape
        and out.dtype == input.dtype
        and input.is_contiguous()
        and other.is_contiguous()
        and out.is_contiguous()
    ):
        n_elem = out.numel()
        if n_elem:
            _launch_flat(input, other, out, n_elem)
        return out

    # ---- general path: validate / resize like the aten out-variant ----
    out_shape = torch.broadcast_shapes(input.shape, other.shape)
    if input.shape[dim] != 3 or other.shape[dim] != 3:
        raise RuntimeError("cross: dimension must be of size 3")
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"cross: expected out dtype {input.dtype}, but got {out.dtype}"
        )
    if out.device != input.device:
        raise RuntimeError("cross: out must be on the same device as input")
    if tuple(out.shape) != tuple(out_shape):
        out.resize_(out_shape)

    sa = _broadcast_strides(input, out_shape)
    sb = _broadcast_strides(other, out_shape)
    so = _broadcast_strides(out, out_shape)

    vectors = out.numel() // 3
    if vectors == 0:
        return out

    blk = 64
    _cross_general_kernel[((vectors + blk - 1) // blk,)](
        input, other, out, vectors, tuple(out_shape), dim,
        tuple(sa), tuple(sb), tuple(so), BLOCK_I=blk)
    return out
