"""cumulative_trapezoid: 融合增量舍入 + 片上扫描的单/双 kernel Triton 候选。

语义：先按输入 dtype 对相邻元素求和并缩放（保留参考实现的低精度舍入），
再沿 dim 扫描；扫描累加 dtype 由 _ACC32 选择（1=fp32 累加，0=输入 dtype 累加）。
"""
import torch
import triton
import triton.language as tl

# 1: 扫描在 fp32 中累加；0: 在输入 dtype 中累加（对齐参考的低精度累加）。
_ACC32 = 1


@triton.jit
def _scan_last(Y, O, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
               ACC32: tl.constexpr):
    """连续末轴：一个 program 处理一行，按 BLOCK 分块并携带标量进位。"""
    N1: tl.constexpr = N - 1
    acc_ty: tl.constexpr = tl.float32 if ACC32 else Y.dtype.element_ty
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    yb = row * N
    ob = row * N1
    if BLOCK >= N1:
        m = k < N1
        a = tl.load(Y + yb + k, mask=m, other=0.0).to(tl.float32)
        b = tl.load(Y + yb + k + 1, mask=m, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty).to(acc_ty)
        s = tl.cumsum(inc, 0)
        tl.store(O + ob + k, s.to(O.dtype.element_ty), mask=m)
    else:
        zero = tl.zeros((BLOCK,), acc_ty)
        carry = zero
        for t in tl.static_range(0, (N1 + BLOCK - 1) // BLOCK):
            off = t * BLOCK + k
            m = off < N1
            a = tl.load(Y + yb + off, mask=m, other=0.0).to(tl.float32)
            b = tl.load(Y + yb + off + 1, mask=m, other=0.0).to(tl.float32)
            inc = ((a + b) * SCALE).to(Y.dtype.element_ty).to(acc_ty)
            s = tl.cumsum(inc, 0) + carry
            tl.store(O + ob + off, s.to(O.dtype.element_ty), mask=m)
            carry = zero + tl.sum(tl.where(k == BLOCK - 1, s, zero), 0)


@triton.jit
def _scan0_tile(Y, O, TOT, SCALE: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
                KB: tl.constexpr, CBW: tl.constexpr, ACC32: tl.constexpr):
    """首轴（行方向）扫描第一阶段：块内局部累加，并记录块总量。"""
    N1: tl.constexpr = N - 1
    acc_ty: tl.constexpr = tl.float32 if ACC32 else Y.dtype.element_ty
    t = tl.program_id(0)
    c = tl.program_id(1)
    rows = t * KB + tl.arange(0, KB)
    cols = c * CBW + tl.arange(0, CBW)
    cmask = cols < M
    mask = (rows < N1)[:, None] & cmask[None, :]
    idx = rows[:, None] * M + cols[None, :]
    a = tl.load(Y + idx, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Y + idx + M, mask=mask, other=0.0).to(tl.float32)
    inc = ((a + b) * SCALE).to(Y.dtype.element_ty).to(acc_ty)
    s = tl.cumsum(inc, 0)
    tl.store(O + idx, s.to(O.dtype.element_ty), mask=mask)
    tot = tl.sum(inc, 0)
    tl.store(TOT + t * M + cols, tot.to(TOT.dtype.element_ty), mask=cmask)


@triton.jit
def _scan0_fix(O, TOT, M: tl.constexpr, N: tl.constexpr, KB: tl.constexpr,
               CBW: tl.constexpr, TMAX: tl.constexpr, ACC32: tl.constexpr):
    """首轴扫描第二阶段：累加前序块总量。"""
    N1: tl.constexpr = N - 1
    acc_ty: tl.constexpr = tl.float32 if ACC32 else O.dtype.element_ty
    t = tl.program_id(0)
    c = tl.program_id(1)
    rows = t * KB + tl.arange(0, KB)
    cols = c * CBW + tl.arange(0, CBW)
    cmask = cols < M
    mask = (rows < N1)[:, None] & cmask[None, :]
    prev = tl.arange(0, TMAX)
    pmask = (prev < t)[:, None] & cmask[None, :]
    p = tl.load(TOT + prev[:, None] * M + cols[None, :], mask=pmask, other=0.0)
    off = tl.sum(p.to(acc_ty), 0)
    idx = rows[:, None] * M + cols[None, :]
    o = tl.load(O + idx, mask=mask, other=0.0)
    res = o.to(acc_ty) + off[None, :]
    tl.store(O + idx, res.to(O.dtype.element_ty), mask=mask)


@triton.jit
def _scan_gen(Y, O, nl, SCALE: tl.constexpr, RSHAPE: tl.constexpr, YST: tl.constexpr,
              OST: tl.constexpr, NDIM: tl.constexpr, S: tl.constexpr,
              OAX: tl.constexpr, M: tl.constexpr, RB: tl.constexpr,
              ACC32: tl.constexpr):
    """任意布局：跨 rest 索引向量化，沿扫描轴串行累加。"""
    acc_ty: tl.constexpr = tl.float32 if ACC32 else Y.dtype.element_ty
    r = tl.program_id(0) * RB + tl.arange(0, RB)
    ok = r < M
    rem = r
    ya = tl.zeros((RB,), tl.int64)
    oa = tl.zeros((RB,), tl.int64)
    for i in tl.static_range(0, NDIM):
        cc = rem % RSHAPE[i]
        rem = rem // RSHAPE[i]
        ya += cc.to(tl.int64) * YST[i]
        oa += cc.to(tl.int64) * OST[i]
    carry = tl.zeros((RB,), acc_ty)
    for i in range(0, nl):
        a = tl.load(Y + ya + i * S, mask=ok, other=0.0).to(tl.float32)
        b = tl.load(Y + ya + (i + 1) * S, mask=ok, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty).to(acc_ty)
        carry = carry + inc
        tl.store(O + oa + i * OAX, carry.to(O.dtype.element_ty), mask=ok)


def _block_for(n1):
    return min(2048, max(1, triton.next_power_of_2(n1)))


def run(y, *, dx=1, dim=-1):
    nd = y.ndim
    if nd == 0 or not -nd <= dim < nd:
        raise IndexError('cumulative_trapezoid: dimension out of range')
    axis = dim % nd
    shape = list(y.shape)
    n = int(shape[axis])
    shape[axis] = max(0, n - 1)
    out = torch.empty(shape, dtype=y.dtype, device=y.device)
    if n < 2 or out.numel() == 0:
        return out
    scale = float(dx) / 2.0
    acc32 = 1 if (y.dtype == torch.float32 or _ACC32) else 0
    n1 = n - 1
    if y.is_contiguous() and axis == nd - 1:
        rows = out.numel() // n1
        _scan_last[(rows,)](y, out, scale, n, _block_for(n1), acc32)
        return out
    if y.is_contiguous() and axis == 0:
        m = out.numel() // n1
        kb = 128
        blocks = -(-m // 32)
        if -(-n1 // kb) * blocks < 16:
            kb = 32 if n1 > 32 else n1
            blocks = -(-m // 16)
        cbw = 32
        nb = -(-m // cbw)
        t = -(-n1 // kb)
        tot = torch.empty((t * m,), dtype=y.dtype, device=y.device)
        _scan0_tile[(t, nb)](y, out, tot, scale, n, m, kb, cbw, acc32)
        _scan0_fix[(t, nb)](out, tot, m, n, kb, cbw, triton.next_power_of_2(t), acc32)
        return out
    ost = list(out.stride())
    dims = [d for d in range(nd) if d != axis]
    rshape = tuple(int(y.shape[d]) for d in reversed(dims))
    yst = tuple(int(y.stride(d)) for d in reversed(dims))
    ostd = tuple(int(ost[d]) for d in reversed(dims))
    m = 1
    for s in rshape:
        m *= s
    rb = min(256, max(16, triton.next_power_of_2(m)))
    _scan_gen[(-(-m // rb),)](y, out, n1, scale, rshape, yst, ostd, len(dims),
                              int(y.stride(axis)), int(ost[axis]), m, rb, acc32)
    return out
