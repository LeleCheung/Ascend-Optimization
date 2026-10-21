# value_selecting_reduction_backward -- Triton implementation.
#
# Semantics (from the reference):
#   out = zeros(sizes); out.scatter_(dim, indices, grad)   (+ unsqueeze when !keepdim)
#
# The non-reduced coordinates of every source element are preserved by the
# scatter, so the mapping  (o, i) -> (o, idx[o, i], i)  is injective and the
# operation is exactly the deterministic elementwise selection
#   out[o, r, i] = grad[o, i] if idx[o, i] == r else 0
# where the output is viewed as (outer, red, inner) around `dim`.
# A single kernel therefore writes the whole output (no separate zero-fill pass).

import torch
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535  # Ascend rtKernelLaunch coreDim limit


def _collapse(shape, stride):
    """Collapse a dim range into (size, stride) if it is affinely strided."""
    n = len(shape)
    if n == 0:
        return 1, 0
    size = 1
    st = 0
    for j in range(n - 1, -1, -1):
        sj = int(shape[j])
        if j == n - 1 or size == 1:
            st = int(stride[j])
            size = sj
        elif sj == 1:
            continue
        elif int(stride[j]) == st * size:
            size *= sj
        else:
            return None
    return size, st


@triton.jit
def _vsrb_inner1_kernel(
    out_ptr,
    g_ptr,
    i_ptr,
    outer,
    red,
    gs_o,
    is_o,
    n_tiles,
    nb,
    BLOCK_R: tl.constexpr,
    EVEN_R: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    r_off = tl.arange(0, BLOCK_R)
    for t in range(pid, n_tiles, nprog):
        o = t // nb
        rb = t % nb
        r = rb * BLOCK_R + r_off
        v = tl.load(i_ptr + o * is_o)
        g = tl.load(g_ptr + o * gs_o)
        val = tl.where(r == v, g, tl.zeros_like(g))
        off = o * red + r
        if EVEN_R:
            tl.store(out_ptr + off, val)
        else:
            tl.store(out_ptr + off, val, mask=r < red)


@triton.jit
def _vsrb_general_kernel(
    out_ptr,
    g_ptr,
    i_ptr,
    n_elem,
    red,
    inner,
    gs_o,
    gs_i,
    is_o,
    is_i,
    n_tiles,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    a = tl.arange(0, BLOCK)
    ri = red * inner
    for t in range(pid, n_tiles, nprog):
        off = t * BLOCK + a
        m = off < n_elem
        o = off // ri
        rem = off % ri
        r = rem // inner
        i = rem % inner
        g = tl.load(g_ptr + o * gs_o + i * gs_i, mask=m, other=0.0)
        v = tl.load(i_ptr + o * is_o + i * is_i, mask=m, other=-1)
        val = tl.where(r == v, g, tl.zeros_like(g))
        tl.store(out_ptr + off, val, mask=m)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def value_selecting_reduction_backward(
    grad: torch.Tensor,
    dim: int,
    indices: torch.Tensor,
    sizes,
    keepdim: bool,
) -> torch.Tensor:
    sizes = [int(s) for s in sizes]
    ndim = len(sizes)
    d = int(dim)
    if d < 0:
        d += ndim

    red = sizes[d]
    outer = 1
    for j in range(d):
        outer *= sizes[j]
    inner = 1
    for j in range(d + 1, ndim):
        inner *= sizes[j]

    out = torch.empty(sizes, dtype=grad.dtype, device=grad.device)
    n_elem = out.numel()
    if n_elem == 0:
        return out

    gshape = list(grad.shape)
    gstride = list(grad.stride())
    ishape = list(indices.shape)
    istride = list(indices.stride())
    if keepdim:
        lo, hi = d, d + 1
    else:
        lo, hi = d, d
    gl = _collapse(gshape[:lo], gstride[:lo])
    gr = _collapse(gshape[hi:], gstride[hi:])
    il = _collapse(ishape[:lo], istride[:lo])
    ir = _collapse(ishape[hi:], istride[hi:])
    if gl is None or gr is None or il is None or ir is None:
        raise NotImplementedError(
            "value_selecting_reduction_backward: unsupported strides"
        )

    gs_o, gs_i = gl[1], gr[1]
    is_o, is_i = il[1], ir[1]

    if inner == 1:
        BLOCK_R = min(_next_pow2(red), 2048)
        nb = triton.cdiv(red, BLOCK_R)
        n_tiles = outer * nb
        grid = (min(n_tiles, _MAX_PROGRAMS),)
        _vsrb_inner1_kernel[grid](
            out,
            grad,
            indices,
            outer,
            red,
            gs_o,
            is_o,
            n_tiles,
            nb,
            BLOCK_R=BLOCK_R,
            EVEN_R=(red % BLOCK_R == 0),
        )
    else:
        BLOCK = 256
        n_tiles = triton.cdiv(n_elem, BLOCK)
        grid = (min(n_tiles, _MAX_PROGRAMS),)
        _vsrb_general_kernel[grid](
            out,
            grad,
            indices,
            n_elem,
            red,
            inner,
            gs_o,
            gs_i,
            is_o,
            is_i,
            n_tiles,
            BLOCK=BLOCK,
        )
    return out


run = value_selecting_reduction_backward
