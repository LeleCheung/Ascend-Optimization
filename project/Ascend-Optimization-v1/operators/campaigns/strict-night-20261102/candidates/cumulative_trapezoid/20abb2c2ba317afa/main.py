"""cumulative_trapezoid: 融合增量舍入 + 扫描（Triton, 单次启动 + 预绑定启动）。

参考语义（同输入实测）：先按输入 dtype 对相邻元素求和并缩放，再沿 dim 做一次
倍增（Hillis-Steele）扫描；fp16/bf16 每步舍入回输入 dtype，fp32 在 fp32 中累加。
本实现保持同样的结合顺序与舍入位置。

分派：
  * 末轴连续：一个 program 一条扫描向量。短行用寄存器 gather 倍增；长行改用
    全局 scratch 的偏移加载（实测比寄存器 gather 快，三 dtype 皆然）。
  * 连续但非末轴：[L, n, M] 视为 M 列、沿行扫描；单次启动内每个 program
    处理一个行块，并在本 program 内重算前序行块的和作为 carry，
    避免额外的转置/前缀 kernel（启动开销是本机主要成本）。
  * 非连续输入：先合并转置成末轴连续，走同一条路径，再写回目标布局。
  * 每个形状组合首次调用后缓存“启动计划”（编译产物 + 参数模板），之后每次
    只做一次字典查找、一次张量分配与一次启动器调用，避开 JIT 派发与重复打包。
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_BLOCK = 4096      # 单块扫描长度上限（UB 约束）
_SCRATCH_MIN = 2048    # 超过该长度改用 scratch 偏移加载
_TILE = 64             # 转置分块
_TILE_ELEMS = 8192     # 单块元素上限（UB 实测）
_NCH_MAX = 16          # 单次启动内允许的最大行块数（冗余前缀成本）
_NCH_TARGET = 8        # 优先把行块数压到该值以内

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


@triton.jit
def _scan_one(Y, O, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
              LOG: tl.constexpr, LOW: tl.constexpr):
    """末轴连续、单块：一个 program 一条扫描向量，寄存器 gather 倍增。"""
    N1: tl.constexpr = N - 1
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    mm = k < N1
    yb = row * N
    ob = row * N1
    a = tl.load(Y + yb + k, mask=mm, other=0.0).to(tl.float32)
    b = tl.load(Y + yb + k + 1, mask=mm, other=0.0).to(tl.float32)
    inc = (a + b) * SCALE
    if LOW:
        inc = inc.to(Y.dtype.element_ty)
    s = inc
    for j in tl.static_range(0, LOG):
        g = tl.gather(s, tl.maximum(k - (1 << j), 0), 0)
        if LOW:
            s = (s + tl.where(k >= (1 << j), g, 0.0).to(Y.dtype.element_ty)).to(
                Y.dtype.element_ty)
        else:
            s = s + tl.where(k >= (1 << j), g, 0.0)
    tl.store(O + ob + k, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _scan_scr(Y, O, SCR, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
              LOG: tl.constexpr, LOW: tl.constexpr):
    """末轴连续、单块：倍增的换位值来自全局 scratch 的偏移加载。"""
    N1: tl.constexpr = N - 1
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    sb = row * BLOCK
    mm = k < N1
    yb = row * N
    ob = row * N1
    a = tl.load(Y + yb + k, mask=mm, other=0.0).to(tl.float32)
    b = tl.load(Y + yb + k + 1, mask=mm, other=0.0).to(tl.float32)
    inc = (a + b) * SCALE
    if LOW:
        s = inc.to(Y.dtype.element_ty)
    else:
        s = inc
    for j in tl.static_range(0, LOG):
        tl.store(SCR + sb + k, s)
        tl.debug_barrier()
        g = tl.load(SCR + sb + k - (1 << j), mask=k >= (1 << j), other=0.0)
        if LOW:
            s = (s + g).to(Y.dtype.element_ty)
        else:
            s = s + g
    tl.store(O + ob + k, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _scan_vec(Y, O, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
              LOG: tl.constexpr, LOW: tl.constexpr, TILES: tl.constexpr):
    """末轴连续、超过单块长度：分块倍增 + 块间 carry。"""
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
        if LOW:
            s = (s + carry).to(Y.dtype.element_ty)
            carry = acc16 + tl.sum(tl.where(k == BLOCK - 1, s, acc16), 0)
        else:
            s = s + carry
            carry = acc32 + tl.sum(tl.where(k == BLOCK - 1, s, acc32), 0)
        tl.store(O + ob + off, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _dim0_scan(Y, O, SCALE: tl.constexpr, N: tl.constexpr, N1: tl.constexpr,
               TC: tl.constexpr, MC: tl.constexpr, M: tl.constexpr,
               NCH: tl.constexpr, LOW: tl.constexpr, CM: tl.constexpr):
    """连续非末轴：单次启动，program 内重算前序行块之和作为 carry。"""
    c = tl.program_id(0)
    cb = tl.program_id(1)
    l = tl.program_id(2)
    t = tl.arange(0, TC)
    m = tl.arange(0, MC)
    col = cb * MC + m
    cok = col < M
    ybase = l * N * M + col[None, :]
    obase = l * N1 * M + col[None, :]
    carry = tl.zeros((MC,), tl.float32)
    carry16 = tl.zeros((MC,), Y.dtype.element_ty)
    for p in tl.static_range(0, NCH):
        idx = p * TC * M + t[:, None] * M
        if p < c:
            if CM:
                va = tl.load(Y + ybase + idx, mask=cok[None, :], other=0.0)
                vb = tl.load(Y + ybase + idx + M, mask=cok[None, :], other=0.0)
            else:
                va = tl.load(Y + ybase + idx)
                vb = tl.load(Y + ybase + idx + M)
            inc = (va.to(tl.float32) + vb.to(tl.float32)) * SCALE
            if LOW:
                inc = inc.to(Y.dtype.element_ty)
                carry16 = carry16 + tl.sum(inc, 0).to(Y.dtype.element_ty)
            else:
                carry = carry + tl.sum(inc, 0)
        elif p == c:
            rok = (p * TC + t) < N1
            msk = rok[:, None] & cok[None, :]
            va = tl.load(Y + ybase + idx, mask=msk, other=0.0)
            vb = tl.load(Y + ybase + idx + M, mask=msk, other=0.0)
            inc = (va.to(tl.float32) + vb.to(tl.float32)) * SCALE
            if LOW:
                s = tl.cumsum(inc.to(Y.dtype.element_ty), 0) + carry16[None, :]
            else:
                s = tl.cumsum(inc, 0) + carry[None, :]
            tl.store(O + obase + idx, s.to(O.dtype.element_ty), mask=msk)


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
    linear=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=linear//N1
    coordinate=linear%N1
    remainder=row
    address=coordinate*AXIS_STRIDE
    for d in tl.static_range(len(RSHAPE)-1,-1,-1):
        address+=(remainder%RSHAPE[d])*OSTRIDES[d]
        remainder=remainder//RSHAPE[d]
    value=tl.load(SRC+linear,linear<ROWS*N1,other=0)
    tl.store(DST+address,value,linear<ROWS*N1)


def _bad_launcher(launcher):
    return (getattr(launcher, 'compile_only', False)
            or getattr(launcher, 'enable_msprof_register_tensor', False)
            or getattr(launcher.metadata, 'debug_enabled', False)
            or getattr(launcher.metadata, 'debug', False))


def _bind(kernel, grid, args, dev):
    """编译一次并返回只更新张量指针的快速启动函数；调试/profiling 模式返回 None。"""
    compiled = kernel[grid](*args)
    launcher = compiled.run
    if _bad_launcher(launcher):
        return None
    nargs = len(args)
    vec = [None] * nargs
    slots = []
    for i, a in enumerate(args):
        if torch.is_tensor(a):
            slots.append(i)
        else:
            vec[i] = a
    pairs = list(enumerate(slots))
    shape3 = tuple(grid) + (1,) * (3 - len(grid))
    launch = launcher.launch
    fn = compiled.function
    pm = compiled.packed_metadata

    def call(tensors):
        for j, s in pairs:
            vec[s] = tensors[j].data_ptr()
        launch(*shape3, _raw_stream(dev), fn, pm, None, None, None, *vec)
    return call


class _Plan:
    __slots__ = ('out_shape', 'dtype', 'device', 'call', 'scratch')

    def __init__(self, out_shape, dtype, device, call, scratch=0):
        self.out_shape = out_shape
        self.dtype = dtype
        self.device = device
        self.call = call
        self.scratch = scratch


_PLANS = {}


def _pow2(n):
    b = triton.next_power_of_2(max(1, n))
    return b, b.bit_length() - 1


def _build(y, out, dx, dim, nd):
    """为可预测的快速路径建立启动计划；无法建立时返回 None（走通用路径）。"""
    axis = dim % nd
    if not y.is_contiguous():
        return None
    n = int(y.shape[axis])
    n1 = n - 1
    scale = float(dx) / 2.0
    low = 1 if y.dtype in (torch.float16, torch.bfloat16) else 0
    dev = y.device.index or 0
    if axis == nd - 1:
        rows = out.numel() // n1
        block, log = _pow2(min(n1, _MAX_BLOCK))
        tiles = -(-n1 // block)
        if tiles > 1:
            call = _bind(_scan_vec, (rows,),
                         (y, out, scale, n, block, log, low, tiles), dev)
            scratch = 0
        elif n1 > _SCRATCH_MIN:
            scratch = rows * block
            scr = torch.empty(scratch, dtype=y.dtype, device=y.device)
            call = _bind(_scan_scr, (rows,),
                         (y, out, scr, scale, n, block, log, low), dev)
        else:
            scratch = 0
            call = _bind(_scan_one, (rows,),
                         (y, out, scale, n, block, log, low), dev)
        if call is None:
            return None
        return _Plan(tuple(out.shape), y.dtype, y.device, call, scratch)
    m_dim = 1
    for d in range(axis + 1, nd):
        m_dim *= int(y.shape[d])
    l_dim = out.numel() // (n1 * m_dim)
    mc = max(8, min(128, triton.next_power_of_2(m_dim)))
    tc = max(1, min(_TILE_ELEMS // mc, triton.next_power_of_2(n1)))
    nch = -(-n1 // tc)
    while nch > _NCH_TARGET and mc > 8:
        mc //= 2
        tc = max(1, min(_TILE_ELEMS // mc, triton.next_power_of_2(n1)))
        nch = -(-n1 // tc)
    if nch > _NCH_MAX:
        return None
    call = _bind(_dim0_scan, (nch, -(-m_dim // mc), l_dim),
                 (y, out, scale, n, n1, tc, mc, m_dim, nch, low,
                  m_dim % mc != 0), dev)
    if call is None:
        return None
    return _Plan(tuple(out.shape), y.dtype, y.device, call, 0)


def _row_scan(y, src, dst, rows, n, n1, scale, low):
    block, log = _pow2(min(n1, _MAX_BLOCK))
    tiles = -(-n1 // block)
    if tiles > 1:
        _scan_vec[(rows,)](src, dst, scale, n, block, log, low, tiles)
    elif n1 > _SCRATCH_MIN:
        scr = torch.empty(rows * block, dtype=y.dtype, device=y.device)
        _scan_scr[(rows,)](src, dst, scr, scale, n, block, log, low)
    else:
        _scan_one[(rows,)](src, dst, scale, n, block, log, low)


def _run_slow(y, dx, dim, out=None):
    """通用路径：非连续输入或 profiling/debug 启动模式。"""
    nd = y.ndim
    axis = dim % nd
    n = int(y.shape[axis])
    n1 = n - 1
    if out is None:
        shape = list(y.shape)
        shape[axis] = n - 1 if n > 0 else 0
        out = torch.empty(shape, dtype=y.dtype, device=y.device)
    if n < 2 or out.numel() == 0:
        return out
    scale = float(dx) / 2.0
    low = 1 if y.dtype in (torch.float16, torch.bfloat16) else 0
    rows = out.numel() // n1
    if y.is_contiguous() and axis == nd - 1:
        _row_scan(y, y, out, rows, n, n1, scale, low)
        return out
    buf = torch.empty(rows * (n + n1), dtype=y.dtype, device=y.device)
    flat = buf[: rows * n]
    tmp = buf[rows * n:]
    dims = [d for d in range(nd) if d != axis]
    _transpose[(-(-n // _TILE), -(-rows // _TILE))](
        y, flat, tuple(int(y.shape[d]) for d in reversed(dims)),
        tuple(int(y.stride(d)) for d in reversed(dims)), len(dims),
        n, rows, int(y.stride(axis)), n, _TILE, _TILE)
    _row_scan(y, flat, tmp, rows, n, n1, scale, low)
    if axis == 0:
        _transpose[(-(-rows // _TILE), -(-n1 // _TILE))](
            tmp,out,(n1,),(1,),1,rows,n1,n1,rows,_TILE,_TILE)
    else:
        _restore[(triton.cdiv(out.numel(),1024),)](
            tmp,out,tuple(int(out.shape[d]) for d in dims),
            tuple(int(out.stride(d)) for d in dims),int(out.stride(axis)),rows,n1,1024)
    return out


def run(y, *, dx=1, dim=-1):
    nd = y.ndim
    if nd == 0 or not -nd <= dim < nd:
        raise IndexError('cumulative_trapezoid: dimension out of range')
    key = (y.shape, y.dtype, dim, y.data_ptr() & 15, dx)
    plan = _PLANS.get(key)
    if plan is None:
        axis = dim % nd
        shape = list(y.shape)
        n = int(shape[axis])
        shape[axis] = n - 1 if n > 0 else 0
        out = torch.empty(shape, dtype=y.dtype, device=y.device)
        if n < 2 or out.numel() == 0:
            _PLANS[key] = _Plan(tuple(shape), y.dtype, y.device, None)
            return out
        plan = _build(y, out, dx, dim, nd)
        if plan is None:
            _PLANS[key] = False
            return _run_slow(y, dx, dim, out)
        _PLANS[key] = plan
        return out
    if plan is False:
        return _run_slow(y, dx, dim)
    out = torch.empty(plan.out_shape, dtype=plan.dtype, device=plan.device)
    if plan.call is None:
        return out
    if plan.scratch:
        scr = torch.empty(plan.scratch, dtype=plan.dtype, device=plan.device)
        plan.call((y, out, scr))
    else:
        plan.call((y, out))
    return out
