"""cumulative_trapezoid: 融合增量舍入 + 倍增（Hillis-Steele）扫描，无转置分派。

参考语义（同输入实测）：先按输入 dtype 对相邻元素求和并缩放，再沿 dim 做一次
倍增扫描；fp16 每步舍入回 fp16，bf16/fp32 在 fp32 中累加，最后写回输入 dtype。

分派：
  * 末轴连续：一个 program 一条扫描向量；块内倍增用寄存器 gather（短行）或
    全局 scratch 偏移加载（长行 fp16，实测更快）。
  * 非末轴连续：二维 [BC, TR] 分块直接按扫描轴步长寻址，块内倍增用寄存器
    gather（轴 1），块间用每列 carry，完全避免整张量转置与其逐元素寻址。
  * 其它布局（非连续）：合并转置成末轴连续后扫描，再转置回目标布局。
  * 所有核函数都走缓存启动器，绕开每次调用的 JIT 派发开销。
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_BLOCK = 4096      # 末轴单块扫描长度上限（UB 约束）
_SCRATCH_MIN = 2048    # 末轴超过该长度改用 scratch 偏移加载
_TILE = 64             # 转置分块（仅非连续回退路径）
_BC = 16               # 非末轴：单 program 覆盖的列数
_TR = 128              # 非末轴：单块沿扫描轴长度

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
def _scan_cols(Y, O, SCALE: tl.constexpr, N: tl.constexpr, COLS: tl.constexpr,
               BC: tl.constexpr, TR: tl.constexpr, NCHUNK: tl.constexpr,
               LOG: tl.constexpr, LOW: tl.constexpr):
    """非末轴连续：二维 [BC, TR] 分块沿扫描轴倍增，逐列 carry。

    元素 (outer o, 位置 k, 列 c) 在 o*N*COLS + k*COLS + c；输出连续。
    """
    NCB: tl.constexpr = (COLS + BC - 1) // BC
    pid = tl.program_id(0)
    o = pid // NCB
    c = (pid % NCB) * BC + tl.arange(0, BC)
    cm = c < COLS
    base = o * N * COLS
    obase = o * (N - 1) * COLS
    jj = tl.arange(0, TR)
    zf = tl.zeros((BC, TR), tl.float32)
    zl = tl.zeros((BC, TR), Y.dtype.element_ty)
    zi = tl.zeros((BC, 1), tl.int32)
    if LOW:
        carry = tl.zeros((BC,), Y.dtype.element_ty)
    else:
        carry = tl.zeros((BC,), tl.float32)
    for t in tl.static_range(0, NCHUNK):
        r = t * TR + jj
        m = (r[None, :] < N - 1) & cm[:, None]
        pin = base + r[None, :] * COLS + c[:, None]
        a = tl.load(Y + pin, mask=m, other=0.0)
        b = tl.load(Y + pin + COLS, mask=(r[None, :] + 1 < N) & cm[:, None], other=0.0)
        inc = ((a.to(tl.float32) + b.to(tl.float32)) * SCALE).to(Y.dtype.element_ty)
        s = inc if LOW else inc.to(tl.float32)
        for j in tl.static_range(0, LOG):
            g = tl.gather(s, tl.maximum(jj - (1 << j), 0)[None, :] + zi, 1)
            keep = (jj >= (1 << j))[None, :]
            if LOW:
                s = (s + tl.where(keep, g, zl)).to(Y.dtype.element_ty)
            else:
                s = s + tl.where(keep, g, zf)
        if LOW:
            s = (s + carry[:, None]).to(Y.dtype.element_ty)
        else:
            s = s + carry[:, None]
        tl.store(O + obase + r[None, :] * COLS + c[:, None],
                 s.to(O.dtype.element_ty), mask=m)
        if NCHUNK > 1:
            last = tl.minimum(TR - 1, N - 2 - t * TR)
            carry = tl.sum(tl.where(jj[None, :] == last, s.to(tl.float32), zf), 1)
            if LOW:
                carry = carry.to(Y.dtype.element_ty)


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


@triton.jit
def _restore(SRC, DST, RSHAPE: tl.constexpr, OSTRIDES: tl.constexpr,
             AXIS_STRIDE: tl.constexpr, ROWS: tl.constexpr, N1: tl.constexpr,
             BLOCK: tl.constexpr):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = linear // N1
    coordinate = linear % N1
    remainder = row
    address = coordinate * AXIS_STRIDE
    for d in tl.static_range(len(RSHAPE) - 1, -1, -1):
        address += (remainder % RSHAPE[d]) * OSTRIDES[d]
        remainder = remainder // RSHAPE[d]
    value = tl.load(SRC + linear, linear < ROWS * N1, other=0)
    tl.store(DST + address, value, linear < ROWS * N1)


_scan_vec_c = _Cached(_scan_vec)
_scan_scr_c = _Cached(_scan_scr)
_transpose_c = _Cached(_transpose)
_restore_c = _Cached(_restore)


def _pow2(n):
    b = triton.next_power_of_2(max(1, n))
    return b, b.bit_length() - 1


def _prods(shape, lo, hi):
    p = 1
    for d in range(lo, hi):
        p *= int(shape[d])
    return p


def _stream_getter(dev_index):
    """返回当前 raw stream 的无参取值器（每次调用都重新查询，不缓存 stream）。"""
    def get():
        return _raw_stream(dev_index)
    return get


class _Plan:
    """按 (shape, dtype, dim, dx, 布局, device) 复用的单核启动计划。

    首次调用走标准 JIT 路径并缓存编译产物；之后每次只做输出分配 + 一次
    原生启动器调用（指针按本次调用重新读取），不缓存任何张量数据或 stream。
    """

    __slots__ = ('kernel', 'grid', 'consts', 'oshape', 'dtype', 'device',
                 'scr_shape', 'launch', 'fn', 'meta', 'g0', 'extra', 'get')

    def __init__(self, kernel, grid, consts, oshape, dtype, device, dev_index,
                 scr_shape=None):
        self.kernel = kernel
        self.grid = grid
        self.consts = consts
        self.oshape = oshape
        self.dtype = dtype
        self.device = device
        self.scr_shape = scr_shape
        self.launch = None
        self.fn = None
        self.meta = None
        self.g0 = tuple(grid) + (1,) * (3 - len(grid))
        self.extra = (None,) * len(consts)
        self.get = _stream_getter(dev_index)

    def _init(self, args):
        compiled = self.kernel[self.grid](*args)
        launcher = compiled.run
        if (getattr(launcher, 'compile_only', False)
                or getattr(launcher, 'enable_msprof_register_tensor', False)
                or getattr(launcher.metadata, 'debug_enabled', False)
                or getattr(launcher.metadata, 'debug', False)):
            self.launch = False
            return
        self.launch = launcher.launch
        self.fn = compiled.function
        self.meta = compiled.packed_metadata

    def __call__(self, y):
        out = torch.empty(self.oshape, dtype=self.dtype, device=self.device)
        if self.scr_shape is None:
            launch = self.launch
            if launch is None:
                self._init((y, out) + self.consts)
            elif launch is False:
                self.kernel[self.grid](y, out, *self.consts)
            else:
                launch(*self.g0, self.get(), self.fn, self.meta, None, None, None,
                       y.data_ptr(), out.data_ptr(), *self.extra)
        else:
            scr = torch.empty(self.scr_shape, dtype=self.dtype, device=self.device)
            launch = self.launch
            if launch is None:
                self._init((y, out, scr) + self.consts)
            elif launch is False:
                self.kernel[self.grid](y, out, scr, *self.consts)
            else:
                launch(*self.g0, self.get(), self.fn, self.meta, None, None, None,
                       y.data_ptr(), out.data_ptr(), scr.data_ptr(), *self.extra)
        return out


class _EmptyPlan:
    """n<2 或空输出：只分配输出。"""

    __slots__ = ('oshape', 'dtype', 'device')

    def __init__(self, oshape, dtype, device):
        self.oshape = oshape
        self.dtype = dtype
        self.device = device

    def __call__(self, y):
        return torch.empty(self.oshape, dtype=self.dtype, device=self.device)


class _LegacyPlan:
    """非连续输入：合并转置 -> 末轴扫描 -> 写回目标布局（通用但较慢）。"""

    __slots__ = ('axis', 'nd', 'dims', 'shape', 'n', 'n1', 'rows', 'low', 'scale',
                 'block', 'log', 'tiles', 'oshape', 'dtype', 'device', 'scr_shape')

    def __init__(self, y, axis, dx, oshape, n, n1):
        self.axis = axis
        self.nd = y.ndim
        self.shape = tuple(int(y.shape[d]) for d in range(y.ndim))
        self.n = n
        self.n1 = n1
        self.oshape = oshape
        self.dtype = y.dtype
        self.device = y.device
        self.dims = [d for d in range(y.ndim) if d != axis]
        self.rows = _prods(oshape, 0, y.ndim) // n1
        self.low = 1 if y.dtype == torch.float16 else 0
        self.scale = float(dx) / 2.0
        self.block, self.log = _pow2(min(n1, _MAX_BLOCK))
        self.tiles = -(-n1 // self.block)
        self.scr_shape = (self.rows * self.block,) if (
            self.low and _SCRATCH_MIN <= n1 <= self.block) else None

    def __call__(self, y):
        axis, n, n1, rows = self.axis, self.n, self.n1, self.rows
        out = torch.empty(self.oshape, dtype=self.dtype, device=self.device)
        buf = torch.empty(rows * (n + n1), dtype=self.dtype, device=self.device)
        flat = buf[: rows * n]
        tmp = buf[rows * n:]
        dims = self.dims
        _transpose_c[(-(-n // _TILE), -(-rows // _TILE))](
            y, flat, tuple(int(y.shape[d]) for d in reversed(dims)),
            tuple(int(y.stride(d)) for d in reversed(dims)), len(dims),
            n, rows, int(y.stride(axis)), n, _TILE, _TILE)
        if self.scr_shape is not None:
            scr = torch.empty(self.scr_shape, dtype=self.dtype, device=self.device)
            _scan_scr_c[(rows,)](flat, tmp, scr, self.scale, n, self.block, self.log)
        else:
            _scan_vec_c[(rows,)](flat, tmp, self.scale, n, self.block, self.log,
                                 self.low, self.tiles)
        if axis == 0:
            _transpose_c[(-(-rows // _TILE), -(-n1 // _TILE))](
                tmp, out, (n1,), (1,), 1, rows, n1, n1, rows, _TILE, _TILE)
        else:
            _restore_c[(triton.cdiv(out.numel(), 1024),)](
                tmp, out, tuple(int(out.shape[d]) for d in dims),
                tuple(int(out.stride(d)) for d in dims), int(out.stride(axis)),
                rows, n1, 1024)
        return out


_PLANS = {}


def _build(y, axis, dx, contig):
    nd = y.ndim
    shape = tuple(int(d) for d in y.shape)
    n = shape[axis]
    oshape = shape[:axis] + ((n - 1) if n > 0 else 0,) + shape[axis + 1:]
    dt = y.dtype
    dev = y.device
    if n < 2 or (0 in oshape):
        return _EmptyPlan(oshape, dt, dev)
    n1 = n - 1
    numel = _prods(oshape, 0, nd)
    if not contig:
        return _LegacyPlan(y, axis, dx, oshape, n, n1)
    low = 1 if dt == torch.float16 else 0
    scale = float(dx) / 2.0
    dev_index = dev.index or 0
    if axis == nd - 1:
        rows = numel // n1
        block, log = _pow2(min(n1, _MAX_BLOCK))
        tiles = -(-n1 // block)
        if low and _SCRATCH_MIN <= n1 <= block:
            return _Plan(_scan_scr, (rows,), (scale, n, block, log),
                         oshape, dt, dev, dev_index, scr_shape=(rows * block,))
        return _Plan(_scan_vec, (rows,), (scale, n, block, log, low, tiles),
                     oshape, dt, dev, dev_index)
    cols = _prods(shape, axis + 1, nd)
    outer = numel // (n1 * cols)
    bc = _BC if cols >= _BC else triton.next_power_of_2(cols)
    tr = min(triton.next_power_of_2(max(1, n1)), _TR)
    log = tr.bit_length() - 1
    nchunk = -(-n1 // tr)
    return _Plan(_scan_cols, (outer * (-(-cols // bc)),),
                 (scale, n, cols, bc, tr, nchunk, log, low), oshape, dt, dev, dev_index)


def run(y, *, dx=1, dim=-1):
    nd = y.ndim
    if nd == 0 or not -nd <= dim < nd:
        raise IndexError('cumulative_trapezoid: dimension out of range')
    axis = dim % nd
    dev_index = y.device.index or 0
    if y.is_contiguous():
        key = (y.shape, y.dtype, axis, float(dx), dev_index, 0)
    else:
        key = (y.shape, y.stride(), y.dtype, axis, float(dx), dev_index, 1)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _build(y, axis, dx, key[-1] == 0)
        _PLANS[key] = plan
    return plan(y)
