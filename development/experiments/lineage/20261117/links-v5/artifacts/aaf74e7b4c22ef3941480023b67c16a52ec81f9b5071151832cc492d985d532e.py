# value_selecting_reduction_backward -- Triton implementation.
#
# Semantics (reference): out = zeros(sizes); out.scatter_(dim, indices, grad)
# (with grad/indices unsqueezed along `dim` when keepdim is False).
#
# The scatter preserves each source element's non-reduced coordinates, so
# (o, i) -> (o, idx[o, i], i) is injective and the whole operation is the
# deterministic elementwise selection
#     out[o, r, i] = grad[o, i] if idx[o, i] == r else 0
# with the output viewed as (outer, red, inner) around `dim`.  A single kernel
# therefore writes the entire output: the reference's separate full-tensor
# zero-fill and scatter launches collapse into one pass.
#
# Ascend notes that shaped this design (measured on target):
#   * rtKernelLaunch rejects more than 65535 blocks -> persistent 1D grid.
#   * A launch costs ~55-70us of host time, so exactly one kernel is launched
#     and the scalar geometry is passed as constexpr to keep the binder cheap.
#   * The cost is dominated by program count, not block size: few programs each
#     streaming large contiguous tiles (8192 elements) reach the fill roofline.

import torch
import triton
import triton.language as tl

_MAX_PROGRAMS = 512  # persistent programs; task-derived upper bound
_TILE_ELEMS = 8192  # elements written per tile (larger tiles exceed the UB)


@triton.jit
def _vsrb_inner1_kernel(
    out_ptr,
    g_ptr,
    i_ptr,
    OUTER: tl.constexpr,
    RED: tl.constexpr,
    GSO: tl.constexpr,
    ISO: tl.constexpr,
    NB: tl.constexpr,
    NT: tl.constexpr,
    BO: tl.constexpr,
    BR: tl.constexpr,
    EVEN_O: tl.constexpr,
    EVEN_R: tl.constexpr,
):
    """inner == 1: out[o, r] = grad[o] if idx[o] == r else 0, contiguous rows."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ao = tl.arange(0, BO)
    ar = tl.arange(0, BR)
    for t in range(pid, NT, nprog):
        o = (t // NB) * BO + ao
        r = (t % NB) * BR + ar
        if EVEN_O:
            v = tl.load(i_ptr + o * ISO).to(tl.int32)
            g = tl.load(g_ptr + o * GSO)
        else:
            mo = o < OUTER
            v = tl.load(i_ptr + o * ISO, mask=mo, other=-1).to(tl.int32)
            g = tl.load(g_ptr + o * GSO, mask=mo, other=0.0)
        g2 = tl.broadcast_to(g[:, None], (BO, BR))
        val = tl.where(r[None, :] == v[:, None], g2, tl.zeros_like(g2))
        off = o[:, None] * RED + r[None, :]
        if EVEN_O and EVEN_R:
            tl.store(out_ptr + off, val)
        elif EVEN_R:
            tl.store(out_ptr + off, val, mask=(o < OUTER)[:, None])
        else:
            tl.store(out_ptr + off, val,
                     mask=(o < OUTER)[:, None] & (r[None, :] < RED))


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
    """inner > 1 (or non-collapsible strides): flat pass over the output."""
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
        v = tl.load(i_ptr + o * is_o + i * is_i, mask=m, other=-1).to(tl.int32)
        val = tl.where(r == v, g, tl.zeros_like(g))
        tl.store(out_ptr + off, val, mask=m)


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


def _lead_stride(shape, stride, d):
    """Collapsed stride of the leading `d` dims, or None when non-affine."""
    if d <= 0:
        return 0
    st = int(stride[d - 1])
    acc = int(shape[d - 1])
    for j in range(d - 2, -1, -1):
        sj = int(shape[j])
        if sj == 1:
            continue
        if acc == 1:
            st = int(stride[j])
            acc = sj
        elif int(stride[j]) == st * acc:
            acc *= sj
        else:
            return None
    return st


def _npow2(n):
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
    nd = len(sizes)
    d = dim
    if d < 0:
        d += nd
    red = int(sizes[d])
    outer = 1
    for j in range(d):
        outer *= int(sizes[j])
    inner = 1
    for j in range(d + 1, nd):
        inner *= int(sizes[j])

    out = torch.empty(sizes, dtype=grad.dtype, device=grad.device)
    if outer * red * inner == 0:
        return out

    gsh, gst = grad.shape, grad.stride()
    ish, ist = indices.shape, indices.stride()
    gso = iso = None
    if inner == 1:
        if keepdim:
            gso = _lead_stride(gsh, gst, d)
            iso = _lead_stride(ish, ist, d)
        else:
            # keepdim=False drops `dim`, so the leading dims coincide.
            gso = _lead_stride(gsh, gst, d)
            iso = _lead_stride(ish, ist, d)

    if gso is not None and iso is not None:
        if red >= _TILE_ELEMS:
            br = _TILE_ELEMS
            bo = 1
        else:
            br = _npow2(red)
            bo = max(1, _TILE_ELEMS // br)
            cap = _npow2(outer)
            if bo > cap:
                bo = cap
        nb = (red + br - 1) // br
        nt = ((outer + bo - 1) // bo) * nb
        _vsrb_inner1_kernel[(nt if nt < _MAX_PROGRAMS else _MAX_PROGRAMS,)](
            out, grad, indices,
            outer, red, gso, iso, nb, nt, bo, br,
            (outer % bo == 0), (red == br),
        )
        return out

    lo, hi = (d, d + 1) if keepdim else (d, d)
    gl = _collapse(gsh[:lo], gst[:lo])
    gr = _collapse(gsh[hi:], gst[hi:])
    il = _collapse(ish[:lo], ist[:lo])
    ir = _collapse(ish[hi:], ist[hi:])
    if gl is None or gr is None or il is None or ir is None:
        raise NotImplementedError(
            "value_selecting_reduction_backward: unsupported strides"
        )
    n_elem = outer * red * inner
    nt = (n_elem + 255) // 256
    _vsrb_general_kernel[(nt if nt < 65535 else 65535,)](
        out, grad, indices,
        n_elem, red, inner, gl[1], gr[1], il[1], ir[1], nt,
        BLOCK=256,
    )
    return out


run = value_selecting_reduction_backward


# Codex追加的独立语义修复：原始KG生成源码保持在上述正文，原始轮次不改写。
@triton.jit
def _hgkda_strided_value_kernel(
    grad, indices, out, N: tl.constexpr, SIZES: tl.constexpr,
    GSTRIDES: tl.constexpr, ISTRIDES: tl.constexpr, DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lanes=tl.arange(0,BLOCK)
    for tile in range(tl.program_id(0),tl.cdiv(N,BLOCK),tl.num_programs(0)):
        offset=tile*BLOCK+lanes
        remaining=offset.to(tl.int64)
        grad_offset=tl.full((BLOCK,),0,tl.int64)
        index_offset=tl.full((BLOCK,),0,tl.int64)
        reduced_coordinate=tl.full((BLOCK,),0,tl.int64)
        for axis in tl.static_range(len(SIZES)-1,-1,-1):
            coordinate=remaining%SIZES[axis]
            remaining=remaining//SIZES[axis]
            if axis==DIM:
                reduced_coordinate=coordinate
            else:
                grad_offset+=coordinate*GSTRIDES[axis]
                index_offset+=coordinate*ISTRIDES[axis]
        value=tl.load(grad+grad_offset,mask=offset<N,other=0.)
        selected=tl.load(indices+index_offset,mask=offset<N,other=-1)
        tl.store(out+offset,tl.where(reduced_coordinate==selected,value,tl.zeros_like(value)),mask=offset<N)


_hgkda_original_value_backward=value_selecting_reduction_backward


def value_selecting_reduction_backward(grad,dim,indices,sizes,keepdim):
    try:
        return _hgkda_original_value_backward(grad,dim,indices,sizes,keepdim)
    except NotImplementedError as error:
        # 只接管已确认的合法stride拒绝，不掩盖编译器或其他运行错误。
        if str(error)!="value_selecting_reduction_backward: unsupported strides":
            raise
    sizes=tuple(int(value) for value in sizes)
    axis=dim%len(sizes)
    grad_strides=list(grad.stride());index_strides=list(indices.stride())
    if keepdim:
        grad_strides[axis]=0;index_strides[axis]=0
    else:
        grad_strides.insert(axis,0);index_strides.insert(axis,0)
    output=torch.empty(sizes,dtype=grad.dtype,device=grad.device)
    elements=output.numel()
    if elements:
        _hgkda_strided_value_kernel[(min(triton.cdiv(elements,256),512),)](
            grad,indices,output,elements,sizes,tuple(grad_strides),tuple(index_strides),axis,256)
    return output


run=value_selecting_reduction_backward
