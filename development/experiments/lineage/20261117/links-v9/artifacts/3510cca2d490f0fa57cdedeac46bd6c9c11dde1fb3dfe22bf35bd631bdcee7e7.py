"""cumulative_trapezoid: 融合增量舍入 + 片上/外挂 Hillis-Steele 倍增扫描。

参考语义（同输入实测）：先按输入 dtype 对相邻元素求和并缩放，再沿 dim 做一次
倍增（Hillis-Steele）扫描；工作精度等于输入 dtype（fp16 每步舍入回 fp16），
bf16/fp32 在 fp32 中累加。本实现保持同样的结合顺序与舍入位置。

分派：
  * 末轴连续：一个 program 一条扫描向量；块内倍增用寄存器 gather（短行）或
    全局 scratch 偏移加载（长行 fp16，实测更快）。
  * 其它布局：先合并转置成末轴连续，走同一条路径，再转置回目标布局，
    避免逐列跨行步长访问造成的延迟暴露。
  * 所有核函数都走缓存启动器，绕开每次调用的 JIT 派发开销。
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_BLOCK = 4096      # 单块扫描长度上限（UB 约束）
_SCRATCH_MIN = 2048    # 超过该长度改用 scratch 偏移加载
_TILE = 64             # 转置分块

_RAW_STREAM = {}


def _raw_stream(device_index):
    getter = _RAW_STREAM.get(device_index)
    if getter is None:
        raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
        expected = int(torch.npu.current_stream(device_index).npu_stream)
        if raw is not None:
            try:
                if int(raw(device_index)) == expected:
                    getter = lambda: raw(device_index)  # noqa: E731
            except TypeError:
                if int(raw()) == expected:
                    getter = raw
        if getter is None:
            from triton.runtime import driver
            getter = lambda: driver.active.get_current_stream(device_index)  # noqa: E731
        _RAW_STREAM[device_index] = getter
    return getter()


class _Cached:
    """首次调用走标准 JIT 路径并缓存编译产物，之后复用启动器与本调用的指针。"""

    def __init__(self, kernel):
        self.kernel = kernel
        self.entries = {}

    def __getitem__(self, grid):
        def invoke(*args):
            key = (tuple(grid),) + tuple(
                (a.data_ptr() & 15, a.shape, a.stride(), a.dtype)
                if torch.is_tensor(a) else a for a in args)
            entry = self.entries.get(key)
            if entry is None:
                compiled = self.kernel[grid](*args)
                launcher = compiled.run
                if (getattr(launcher, 'compile_only', False)
                        or getattr(launcher, 'enable_msprof_register_tensor', False)
                        or getattr(launcher.metadata, 'debug_enabled', False)
                        or getattr(launcher.metadata, 'debug', False)):
                    self.entries[key] = False
                    return
                shape = tuple(grid) + (1,) * (3 - len(grid))
                self.entries[key] = (launcher.launch, compiled.function,
                                     compiled.packed_metadata, shape)
                return
            if entry is False:
                self.kernel[grid](*args)
                return
            ptrs = tuple(a.data_ptr() if torch.is_tensor(a) else None for a in args)
            entry[0](*entry[3], _raw_stream(args[0].device.index or 0),
                     entry[1], entry[2], None, None, None, *ptrs)
        return invoke


@triton.jit
def _scan_vec(Y, O, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
              LOG: tl.constexpr, LOW: tl.constexpr, TILES: tl.constexpr):
    """末轴连续：一个 program 一条扫描向量，寄存器内倍增。"""
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
def _scan_scr(Y, O, SCR, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
              LOG: tl.constexpr):
    """末轴连续、低精度累加：倍增的换位值来自全局 scratch 的偏移加载。"""
    N1: tl.constexpr = N - 1
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    yb = row * N
    ob = row * N1
    sb = row * BLOCK
    mm = k < N1
    a = tl.load(Y + yb + k, mask=mm, other=0.0).to(tl.float32)
    b = tl.load(Y + yb + k + 1, mask=mm, other=0.0).to(tl.float32)
    inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
    s = inc
    for j in tl.static_range(0, LOG):
        tl.store(SCR + sb + k, s)
        tl.debug_barrier()
        g = tl.load(SCR + sb + k - (1 << j), mask=k >= (1 << j), other=0.0)
        s = (s + g).to(Y.dtype.element_ty)
    tl.store(O + ob + k, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _transpose(SRC, DST, RSHAPE: tl.constexpr, YST: tl.constexpr, NDIM: tl.constexpr,
               R: tl.constexpr, C: tl.constexpr, S: tl.constexpr, DA: tl.constexpr,
               RB: tl.constexpr, CB: tl.constexpr):
    """把 [R, C] 扫描矩阵转置搬入连续缓冲：读写两侧都沿最内轴连续。"""
    r = tl.program_id(0) * RB + tl.arange(0, RB)
    c = tl.program_id(1) * CB + tl.arange(0, CB)
    rem = c
    ya = tl.zeros((CB,), tl.int32)
    for i in tl.static_range(0, NDIM):
        cc = rem % RSHAPE[i]
        rem = rem // RSHAPE[i]
        ya += cc * YST[i]
    rm = r < R
    cm = c < C
    tile = tl.load(SRC + ya[None, :] + r[:, None] * S,
                   mask=rm[:, None] & cm[None, :], other=0.0)
    tl.store(DST + c[:, None] * DA + r[None, :], tl.trans(tile),
             mask=cm[:, None] & rm[None, :])


_scan_vec = _Cached(_scan_vec)
_scan_scr = _Cached(_scan_scr)
_transpose = _Cached(_transpose)


def _pow2(n):
    b = triton.next_power_of_2(max(1, n))
    return b, b.bit_length() - 1


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
    block, log = _pow2(min(n1, _MAX_BLOCK))
    tiles = -(-n1 // block)
    rows = out.numel() // n1
    if y.is_contiguous() and axis == nd - 1:
        if low and n1 >= _SCRATCH_MIN:
            scr = torch.empty(rows * block, dtype=y.dtype, device=y.device)
            _scan_scr[(rows,)](y, out, scr, scale, n, block, log)
        else:
            _scan_vec[(rows,)](y, out, scale, n, block, log, low, tiles)
        return out
    buf = torch.empty(rows * (n + n1), dtype=y.dtype, device=y.device)
    flat = buf[: rows * n]
    tmp = buf[rows * n:]
    dims = [d for d in range(nd) if d != axis]
    _transpose[(-(-n // _TILE), -(-rows // _TILE))](
        y, flat, tuple(int(y.shape[d]) for d in reversed(dims)),
        tuple(int(y.stride(d)) for d in reversed(dims)), len(dims),
        n, rows, int(y.stride(axis)), n, _TILE, _TILE)
    if low and n1 >= _SCRATCH_MIN:
        scr = torch.empty(rows * block, dtype=y.dtype, device=y.device)
        _scan_scr[(rows,)](flat, tmp, scr, scale, n, block, log)
    else:
        _scan_vec[(rows,)](flat, tmp, scale, n, block, log, low, tiles)
    _transpose[(-(-rows // _TILE), -(-n1 // _TILE))](
        tmp, out, (rows,), (n1,), 1, rows, n1, 1, rows, _TILE, _TILE)
    return out
