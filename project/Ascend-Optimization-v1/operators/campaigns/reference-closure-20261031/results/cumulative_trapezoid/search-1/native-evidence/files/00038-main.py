"""cumulative_trapezoid: 融合增量舍入 + 片上 Hillis-Steele 倍增扫描。

参考语义（同输入实测）：先按输入 dtype 对相邻元素求和并缩放，再沿 dim 做一次
倍增（Hillis-Steele）扫描；工作精度等于输入 dtype（fp16 每步舍入回 fp16），
bf16/fp32 在 fp32 中累加。本实现保持同样的结合顺序与舍入位置。

布局分派：
  * 末轴连续：每条扫描向量一个 program，一维倍增（访存完全连续）。
  * 其它布局：二维块沿扫描轴倍增，块内保持全长以减少跨块进位。
"""
import torch
import triton
import triton.language as tl

_MAX_VEC_BLOCK = 4096   # 一维扫描块上限（UB 约束）
_MAX_MAT_KB = 1024      # 二维扫描块上限
_MAT_BUDGET = 512       # 二维块元素预算，控制 UB 占用


def _pow2(n):
    b = triton.next_power_of_2(max(1, n))
    return b, b.bit_length() - 1


@triton.jit
def _scan_vec(Y, O, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
              LOG: tl.constexpr, LOW: tl.constexpr, TILES: tl.constexpr):
    """末轴连续：一个 program 处理一条扫描向量，一维倍增 + 标量进位。"""
    N1: tl.constexpr = N - 1
    acc16 = tl.zeros((BLOCK,), Y.dtype.element_ty)
    acc32 = tl.zeros((BLOCK,), tl.float32)
    k = tl.arange(0, BLOCK)
    row = tl.program_id(0)
    yb = row * N
    ob = row * N1
    carry = acc16 if LOW else acc32
    for t in tl.static_range(0, TILES):
        off = t * BLOCK + k
        mm = off < N1
        a = tl.load(Y + yb + off, mask=mm, other=0.0).to(tl.float32)
        b = tl.load(Y + yb + off + 1, mask=mm, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
        s = inc if LOW else inc.to(tl.float32)
        for j in tl.static_range(0, LOG):
            g = tl.gather(s, tl.maximum(k - (1 << j), 0), 0)
            if LOW:
                s = (s + tl.where(k >= (1 << j), g, acc16)).to(Y.dtype.element_ty)
            else:
                s = s + tl.where(k >= (1 << j), g, acc32)
        if TILES > 1:
            if LOW:
                s = (s + carry).to(Y.dtype.element_ty)
                carry = acc16 + tl.sum(tl.where(k == BLOCK - 1, s, acc16), 0)
            else:
                s = s + carry
                carry = acc32 + tl.sum(tl.where(k == BLOCK - 1, s, acc32), 0)
        tl.store(O + ob + off, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _scan_mat(Y, O, SCALE: tl.constexpr, RSHAPE: tl.constexpr, YST: tl.constexpr,
              OST: tl.constexpr, NDIM: tl.constexpr, S: tl.constexpr, OAX: tl.constexpr,
              N: tl.constexpr, M: tl.constexpr, KB: tl.constexpr, LOG: tl.constexpr,
              CBW: tl.constexpr, LOW: tl.constexpr, TILES: tl.constexpr):
    """非末轴布局：块内沿扫描轴倍增，跨块按列向量进位。"""
    N1: tl.constexpr = N - 1
    kb = tl.arange(0, KB)
    kbi = tl.broadcast_to(kb[:, None], (KB, CBW))
    col = tl.program_id(0) * CBW + tl.arange(0, CBW)
    ok = col < M
    rem = col
    ya = tl.zeros((CBW,), tl.int32)
    oa = tl.zeros((CBW,), tl.int32)
    for i in tl.static_range(0, NDIM):
        cc = rem % RSHAPE[i]
        rem = rem // RSHAPE[i]
        ya += cc * YST[i]
        oa += cc * OST[i]
    acc16 = tl.zeros((KB, CBW), Y.dtype.element_ty)
    acc32 = tl.zeros((KB, CBW), tl.float32)
    c16 = tl.zeros((CBW,), Y.dtype.element_ty)
    c32 = tl.zeros((CBW,), tl.float32)
    carry = c16 if LOW else c32
    for t in tl.static_range(0, TILES):
        rows = t * KB + kb
        mm = (rows < N1)[:, None] & ok[None, :]
        idx = ya[None, :] + rows[:, None] * S
        a = tl.load(Y + idx, mask=mm, other=0.0).to(tl.float32)
        b = tl.load(Y + idx + S, mask=mm, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
        s = inc if LOW else inc.to(tl.float32)
        for j in tl.static_range(0, LOG):
            g = tl.gather(s, tl.maximum(kbi - (1 << j), 0), 0)
            if LOW:
                s = (s + tl.where(kbi >= (1 << j), g, acc16)).to(Y.dtype.element_ty)
            else:
                s = s + tl.where(kbi >= (1 << j), g, acc32)
        if TILES > 1:
            if LOW:
                s = (s + carry[None, :]).to(Y.dtype.element_ty)
                carry = c16 + tl.sum(tl.where(kbi == KB - 1, s, acc16), 0)
            else:
                s = s + carry[None, :]
                carry = c32 + tl.sum(tl.where(kbi == KB - 1, s, acc32), 0)
        tl.store(O + oa[None, :] + rows[:, None] * OAX, s.to(O.dtype.element_ty), mask=mm)


def run(y, *, dx=1, dim=-1):
    nd = y.ndim
    if nd == 0 or not -nd <= dim < nd:
        raise IndexError('cumulative_trapezoid: dimension out of range')
    axis = dim % nd
    shape = list(y.shape)
    n = int(shape[axis])
    shape[axis] = n - 1 if n > 0 else 0
    out = torch.empty(shape, dtype=y.dtype, device=y.device)
    if n < 2 or out.numel() == 0:
        return out
    scale = float(dx) / 2.0
    low = 1 if y.dtype == torch.float16 else 0
    n1 = n - 1
    if y.is_contiguous() and axis == nd - 1:
        block, log = _pow2(min(n1, _MAX_VEC_BLOCK))
        _scan_vec[(out.numel() // n1,)](y, out, scale, n, block, log, low,
                                        -(-n1 // block))
        return out
    dims = [d for d in range(nd) if d != axis]
    m = out.numel() // n1
    kb, log = _pow2(min(n1, _MAX_MAT_KB))
    per = max(1, _MAT_BUDGET // kb)
    cbw = min(triton.next_power_of_2(m), 64, triton.next_power_of_2(per))
    _scan_mat[(-(-m // cbw),)](
        y, out, scale,
        tuple(int(y.shape[d]) for d in reversed(dims)),
        tuple(int(y.stride(d)) for d in reversed(dims)),
        tuple(int(out.stride(d)) for d in reversed(dims)),
        len(dims), int(y.stride(axis)), int(out.stride(axis)),
        n, m, kb, log, cbw, low, -(-n1 // kb))
    return out
