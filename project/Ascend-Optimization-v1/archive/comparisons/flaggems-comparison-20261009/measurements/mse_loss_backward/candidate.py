"""Triton implementation of ``mse_loss_backward`` for Ascend NPU.

Semantics (matching ``torch.ops.aten.mse_loss_backward``):

    out = norm * grad_output * (self - target)
    norm = 2 / self.numel()   if reduction == 1 (mean)
    norm = 2                  otherwise (none / sum)

The multiply order ``(norm * grad_output) * (self - target)`` reproduces the
reference rounding order, so fp32 results agree with ATen to within one
rounding step and fp16 / bf16 results are rounded once out of a full fp32
computation.

Layout of the implementation:

* Hot path - one flat elementwise kernel over the common case where
  ``grad_output``, ``self`` and ``target`` share a shape and are contiguous.
  The block size is derived from the element count only (never from a physical
  core count), so the launch is portable across 910B variants.
* The kernel is launched through the compiled-kernel entry point once it has
  been built for the exact argument specialization, which removes the per-call
  Python re-binding cost of the Triton dispatcher.  Every specialization is
  first launched through the regular dispatcher, so a cached entry is always a
  compiled kernel that has already produced a correct result; any failure falls
  back to the dispatcher path for that specialization.
* Cold path - a strided / broadcasting fallback keeps the operator correct for
  every other input layout (scalar gradients, broadcast operands, non
  contiguous views, arbitrary rank).
"""

import torch
import triton
import triton.language as tl

_BIG_BLOCK = 8192

_REDUCTION_BY_NAME = {"none": 0, "mean": 1, "sum": 2}

# specialization key -> (c_launcher, function, packed_metadata) or False (disabled)
_LAUNCH_CACHE = {}

_DIRECT = {"resolved": False, "device": None, "stream": None}


@triton.jit
def _mse_bwd_flat(g_ptr, s_ptr, t_ptr, o_ptr, factor, n,
                  BLOCK: tl.constexpr, MASK: tl.constexpr):
    """out[i] = factor * grad[i] * (self[i] - target[i]) over a flat range."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if MASK:
        m = off < n
        gv = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + off, mask=m, other=0.0).to(tl.float32)
        tv = tl.load(t_ptr + off, mask=m, other=0.0).to(tl.float32)
        r = (factor * gv) * (sv - tv)
        tl.store(o_ptr + off, r.to(o_ptr.dtype.element_ty), mask=m)
    else:
        gv = tl.load(g_ptr + off).to(tl.float32)
        sv = tl.load(s_ptr + off).to(tl.float32)
        tv = tl.load(t_ptr + off).to(tl.float32)
        r = (factor * gv) * (sv - tv)
        tl.store(o_ptr + off, r.to(o_ptr.dtype.element_ty))


@triton.jit
def _mse_bwd_affine(g_ptr, s_ptr, t_ptr, o_ptr, factor, total, isize,
                    g_row, g_col, s_row, s_col, t_row, t_col,
                    BLOCK: tl.constexpr, MASK: tl.constexpr):
    """Elementwise pass over the flattened (outer, inner) view of a broadcast set.

    Every operand address is affine in ``(outer, inner)``: ``row`` is its step per
    outer index and ``col`` its step per inner index (0 where that operand is
    broadcast along the dimension).  The output is written contiguously.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    j = idx % isize
    o = idx // isize
    go = o * g_row + j * g_col
    so = o * s_row + j * s_col
    to = o * t_row + j * t_col
    if MASK:
        m = idx < total
        gv = tl.load(g_ptr + go, mask=m, other=0.0).to(tl.float32)
        sv = tl.load(s_ptr + so, mask=m, other=0.0).to(tl.float32)
        tv = tl.load(t_ptr + to, mask=m, other=0.0).to(tl.float32)
        r = (factor * gv) * (sv - tv)
        tl.store(o_ptr + idx, r.to(o_ptr.dtype.element_ty), mask=m)
    else:
        gv = tl.load(g_ptr + go).to(tl.float32)
        sv = tl.load(s_ptr + so).to(tl.float32)
        tv = tl.load(t_ptr + to).to(tl.float32)
        r = (factor * gv) * (sv - tv)
        tl.store(o_ptr + idx, r.to(o_ptr.dtype.element_ty))


def _resolve_direct():
    """Resolve the device/stream accessors used by the backend launcher."""
    ctx = _DIRECT
    device = None
    stream = None
    try:
        import torch_npu

        _c = torch_npu._C
        get_dev = getattr(_c, "_npu_getDevice", None)
        get_str = getattr(_c, "_npu_getCurrentRawStreamNoWait", None)
        if get_dev is not None and get_str is not None:
            device, stream = get_dev, get_str
    except BaseException:
        device = stream = None
    if stream is None:
        try:
            from triton.runtime import driver as _drv

            device = _drv.active.get_current_device
            stream = _drv.active.get_current_stream
        except BaseException:
            device = stream = None
    ctx["device"] = device
    ctx["stream"] = stream
    ctx["resolved"] = True


def _build_direct(g, s, t, o, factor, n, blk, mask, grid):
    """Compile/lookup the kernel for this specialization and return a raw launcher."""
    try:
        ck = _mse_bwd_flat.warmup(g, s, t, o, factor, n,
                                  BLOCK=blk, MASK=mask, grid=grid)
        ck._init_handles()
        return ck._run.launch, ck.function, ck.packed_metadata
    except BaseException:
        return False


def _prod(shape):
    p = 1
    for d in shape:
        p *= d
    return p


def _affine(shape, stride):
    """(outer, inner, row_step, col_step) if every address is affine in (o, j)."""
    rank = len(shape)
    if rank == 0:
        return 1, 1, 0, 0
    isize = shape[-1]
    osize = _prod(shape[:-1]) if rank > 1 else 1
    if isize == 0 or osize == 0:
        return None
    col = stride[-1]
    if rank == 1:
        return osize, isize, 0, col
    row = stride[-2]
    span = 1
    for k in range(rank - 2, -1, -1):
        if stride[k] != row * span:
            return None
        if k:
            span *= shape[k]
    return osize, isize, row, col


def _emit(g, s, t, out_flat, off, shape, factor):
    """Launch work for one (sub-)problem whose operands all have ``shape``."""
    total = _prod(shape)
    if total == 0:
        return
    a_g = _affine(shape, g.stride())
    a_s = _affine(shape, s.stride())
    a_t = _affine(shape, t.stride())
    if a_g is not None and a_s is not None and a_t is not None:
        blk = _BIG_BLOCK if total > _BIG_BLOCK else triton.next_power_of_2(total)
        mask = (total % blk) != 0
        grid = ((total + blk - 1) // blk,)
        view = out_flat if off == 0 else out_flat.narrow(0, off, total)
        _mse_bwd_affine[grid](
            g, s, t, view, factor, total, a_g[1],
            a_g[2], a_g[3], a_s[2], a_s[3], a_t[2], a_t[3],
            BLOCK=blk, MASK=mask,
        )
        return
    # Non-affine broadcast pattern: peel the leading dimension (a size-1 slice is
    # dropped, lowering the rank) and write each slice into its slice of the
    # contiguous output buffer.
    sub = shape[1:]
    sub_n = _prod(sub)
    for k in range(shape[0]):
        _emit(g[k], s[k], t[k], out_flat, off + k * sub_n, sub, factor)


def _generic(grad_output, self, target, red):
    gb, sb, tb = torch.broadcast_tensors(grad_output, self, target)
    shape = gb.shape
    total = _prod(shape)
    out = torch.empty(shape, dtype=self.dtype, device=self.device)
    if total:
        # ATen scales by the *self* operand's element count, not by the
        # broadcast (output) size.
        factor = 2.0 / self.numel() if red == 1 else 2.0
        _emit(gb, sb, tb, out.view(-1), 0, shape, factor)
    return out


def _coerce_reduction(reduction):
    if isinstance(reduction, str):
        return _REDUCTION_BY_NAME.get(reduction.lower(), 1)
    try:
        return int(reduction)
    except Exception:
        return 1


def run(grad_output, self, target, reduction=1):
    red = reduction if type(reduction) is int else _coerce_reduction(reduction)

    shape = self.shape
    if (shape == target.shape and shape == grad_output.shape
            and self.is_contiguous() and target.is_contiguous()
            and grad_output.is_contiguous()):
        n = self.numel()
        out = torch.empty_like(self)
        if n:
            factor = 2.0 / n if red == 1 else 2.0
            if n <= _BIG_BLOCK:
                blk = triton.next_power_of_2(n)
                mask = blk != n
                grid = (1,)
            else:
                blk = _BIG_BLOCK
                mask = (n % _BIG_BLOCK) != 0
                grid = ((n + _BIG_BLOCK - 1) // _BIG_BLOCK,)
            ctx = _DIRECT
            if not ctx["resolved"]:
                _resolve_direct()
            dev_fn = ctx["device"]
            dev = None
            if dev_fn is not None:
                try:
                    dev = dev_fn()
                except BaseException:
                    dev = None
            if dev is not None and ((grad_output.data_ptr() | self.data_ptr()
                                     | target.data_ptr() | out.data_ptr()) & 15) == 0:
                key = (dev, grad_output.dtype, self.dtype, target.dtype, out.dtype,
                       blk, mask, n)
                entry = _LAUNCH_CACHE.get(key)
                if entry is None:
                    # First time this specialization is seen: go through the
                    # regular dispatcher (compiles + launches), then cache the
                    # compiled kernel handle for the following calls.
                    _mse_bwd_flat[grid](grad_output, self, target, out, factor, n,
                                        BLOCK=blk, MASK=mask)
                    _LAUNCH_CACHE[key] = _build_direct(
                        grad_output, self, target, out, factor, n, blk, mask, grid)
                    return out
                if entry is not False:
                    stream_fn = ctx["stream"]
                    if stream_fn is not None:
                        try:
                            stream = stream_fn(dev)
                            entry[0](grid[0], 1, 1, stream, entry[1], entry[2],
                                     None, None, None,
                                     grad_output, self, target, out, factor, n,
                                     None, None)
                            return out
                        except BaseException:
                            _LAUNCH_CACHE[key] = False
            _mse_bwd_flat[grid](grad_output, self, target, out, factor, n,
                                BLOCK=blk, MASK=mask)
        return out

    return _generic(grad_output, self, target, red)
