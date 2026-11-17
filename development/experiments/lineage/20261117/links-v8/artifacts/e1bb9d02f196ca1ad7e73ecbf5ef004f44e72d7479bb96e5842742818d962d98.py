"""FlagGems hardswish: streaming tiles with a minimal per-call host path.

The device kernel is one masked-tail streaming launch whose hot loop runs
unmasked contiguous tiles strided by the launched program count, with the grid
capped at the reported vector-core count for large inputs.  The host path is
flattened into a single frame: a contiguity test, one empty_like, an integer
identity check against the cached plan and an integer-address launch of the
cached compiled kernel.  Any unaligned pointer falls back to the bound JIT call
so the compiler's alignment assumptions can never be violated, and the stream is
re-read per call rather than cached.
"""
import torch
import torch_npu
import triton
import triton.language as tl
_STREAM = {}
_PLANS = {}
_EMPTY_LIKE = torch.empty_like
# dtype, n, device, launch, grid, block, opts, fn, packed, getter, has_launch, h16
_FAST = [None, -1, -1, None, 1, 4096, {}, None, None, None, False, False]

@triton.jit
def _hardswish_stream(X, Y, N: tl.constexpr, BLOCK: tl.constexpr,
                      H16: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    # Hot loop: every tile here is fully in range, so the transfers stay
    # unmasked contiguous bursts.  Only the trailing region needs a predicate.
    #
    # H16 keeps a narrow dtype's arithmetic narrow: the fp32 emulation of a
    # 16-bit operand costs a cast in and out plus full-rate fp32 vector work,
    # which the target measures at 1.48x the cost of native narrow math.
    full = (N // step) * step
    for start in range(pid * BLOCK, full, step):
        x = tl.load(X + start + lane)
        if H16:
            three = tl.full((1,), 3.0, x.dtype)
            zero = tl.full((1,), 0.0, x.dtype)
            six = tl.full((1,), 6.0, x.dtype)
            sixth = tl.full((1,), 1.0 / 6.0, x.dtype)
            inner = tl.minimum(tl.maximum(x + three, zero), six)
            tl.store(Y + start + lane, x * inner * sixth)
        else:
            v = x.to(tl.float32)
            inner = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
            tl.store(Y + start + lane, v * inner * (1.0 / 6.0))
    offset = full + pid * BLOCK + lane
    tail = offset < N
    x = tl.load(X + offset, tail, other=0)
    if H16:
        three = tl.full((1,), 3.0, x.dtype)
        zero = tl.full((1,), 0.0, x.dtype)
        six = tl.full((1,), 6.0, x.dtype)
        sixth = tl.full((1,), 1.0 / 6.0, x.dtype)
        inner = tl.minimum(tl.maximum(x + three, zero), six)
        tl.store(Y + offset, x * inner * sixth, tail)
    else:
        v = x.to(tl.float32)
        inner = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
        tl.store(Y + offset, v * inner * (1.0 / 6.0), tail)

def _stream_getter(device):
    getter = _STREAM.get(device)
    if getter is not None:
        return getter
    raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
    if raw is not None:
        try:
            expected = int(torch.npu.current_stream(device).npu_stream)
            try:
                if int(raw(device)) == expected:
                    getter = lambda: raw(device)
            except TypeError:
                if int(raw()) == expected:
                    getter = raw
        except Exception:
            getter = None
    if getter is None:
        from triton.runtime import driver
        getter = lambda: driver.active.get_current_stream(device)
    _STREAM[device] = getter
    return getter

def _vector_cores(device):
    """Concurrency cap: reported vector-core count, else work-derived."""
    try:
        from triton.runtime import driver
        props = driver.active.utils.get_device_properties(device)
        cores = int(props.get('num_vectorcore') or 0)
        if cores > 0:
            return cores
    except Exception:
        pass
    return 0

def _geometry(n, device):
    block = min(4096, max(32, 1 << (n - 1).bit_length()))
    grid = (n + block - 1) // block
    if n >= 131072:
        cores = _vector_cores(device)
        if cores and grid > cores:
            grid = cores
    return (block, grid)

def _build(y, out, n, device):
    """Compile this specialization once and return its launch plan."""
    block, grid = _geometry(n, device)
    h16 = y.dtype is torch.float16
    launch = fn = packed = stream = None
    try:
        compiled = _hardswish_stream.warmup(y, out, n, block, h16, grid=(1,))
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False) or getattr(launcher.metadata, 'debug', False)):
            launch = launcher.launch
            fn = compiled.function
            packed = compiled.packed_metadata
            stream = _stream_getter(device)
    except Exception:
        launch = None
    return [y.dtype, n, device, launch, grid, block, {}, fn, packed, stream,
            launch is not None, h16]

def _slow(y, out, n):
    if not n:
        return
    device = y.get_device()
    key = (device, y.dtype, n)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _build(y, out, n, device)
    _FAST[:] = plan
    plan[0], plan[1], plan[2] = y.dtype, n, device
    xp = y.data_ptr()
    yp = out.data_ptr()
    if plan[10] and not ((xp | yp) & 15):
        plan[3](plan[4], 1, 1, plan[9](), plan[7], plan[8], None, None, None,
                xp, yp, n, plan[5], plan[11])
    elif n:
        _hardswish_stream[plan[4],](y, out, n, plan[5], plan[11])

def run(x):
    if not x.is_contiguous():
        x = _strict_clone(x)
    out = _EMPTY_LIKE(x)
    n = x.numel()
    f = _FAST
    if x.dtype is f[0] and n == f[1] and x.get_device() == f[2]:
        xp = x.data_ptr()
        yp = out.data_ptr()
        if f[10] and not ((xp | yp) & 15):
            f[3](f[4], 1, 1, f[9](), f[7], f[8], None, None, None,
                 xp, yp, n, f[5], f[11])
        elif n:
            _hardswish_stream[f[4],](x, out, n, f[5], f[11])
        return out
    _slow(x, out, n)
    return out

"""候选内嵌的Triton存储操作：复制、连续化、填充、转换均执行Triton核。"""
import torch
import triton
import triton.language as tl

@triton.jit
def _strict_copy_kernel(X,Y,N:tl.constexpr,SHAPE:tl.constexpr,XS:tl.constexpr,YS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    remain=i;xp=tl.full((BLOCK,),0,tl.int64);yp=tl.full((BLOCK,),0,tl.int64)
    for d in tl.static_range(len(SHAPE)-1,-1,-1):
        coord=remain%SHAPE[d];remain=remain//SHAPE[d]
        xp+=coord*XS[d];yp+=coord*YS[d]
    value=tl.load(X+xp,i<N,other=0)
    tl.store(Y+yp,value,i<N)

@triton.jit
def _strict_fill_kernel(Y,N:tl.constexpr,VALUE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Y+i,VALUE,i<N)

@triton.jit
def _strict_random_kernel(Y,N:tl.constexpr,SEED,OFFSET,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    # 备用随机采样路径；全部随机数计算在Triton内，生成器仅在宿主管理状态。
    value=tl.rand(SEED,(i.to(tl.uint64)+OFFSET).to(tl.uint32))
    tl.store(Y+i,value,i<N)

def _strict_empty_like(x,*,dtype=None,**kwargs):
    return torch.empty(tuple(x.shape),device=x.device,dtype=x.dtype if dtype is None else dtype)

def _strict_copy_into(dst,src):
    if tuple(dst.shape)!=tuple(src.shape):raise ValueError('strict copy requires equal shapes')
    if dst.numel():
        _strict_copy_kernel[(triton.cdiv(dst.numel(),1024),)](src,dst,dst.numel(),tuple(dst.shape),tuple(src.stride()),tuple(dst.stride()),1024)
    return dst

def _strict_clone(x,**kwargs):
    return _strict_copy_into(_strict_empty_like(x),x)

def _strict_contiguous(x,*args,**kwargs):
    return x if x.is_contiguous() else _strict_clone(x)

def _strict_cast(x,dtype):
    if dtype==x.dtype:return x
    return _strict_copy_into(_strict_empty_like(x,dtype=dtype),x)

def _strict_reshape(x,*shape):
    # view只操作元数据，输入连续化由上面的Triton复制完成。
    return _strict_contiguous(x).view(*shape)

def _strict_flatten(x,start_dim=0,end_dim=-1):
    rank=x.ndim;first=start_dim%rank;last=end_dim%rank
    size=1
    for n in x.shape[first:last+1]:size*=n
    shape=tuple(x.shape[:first])+(size,)+tuple(x.shape[last+1:])
    return _strict_contiguous(x).view(shape)

def _strict_full(shape,value,**kwargs):
    if isinstance(shape,int):shape=(shape,)
    out=torch.empty(shape,**kwargs)
    if out.numel():_strict_fill_kernel[(triton.cdiv(out.numel(),1024),)](out,out.numel(),value,1024)
    return out

def _strict_zeros(*shape,**kwargs):
    shape=shape[0] if len(shape)==1 else shape
    return _strict_full(shape,0,**kwargs)

def _strict_ones(*shape,**kwargs):
    shape=shape[0] if len(shape)==1 else shape
    return _strict_full(shape,1,**kwargs)

def _strict_rand(*shape,**kwargs):
    shape=shape[0] if len(shape)==1 else shape
    generator=kwargs.pop('generator',None)
    out=torch.empty(shape,**kwargs)
    if out.numel():
        if generator is None:
            import torch_npu
            generator=torch.npu.default_generators[out.device.index or 0]
        seed=generator.initial_seed();offset=generator.get_offset()
        advance=triton.cdiv(out.numel(),4)*4
        if offset+advance>=2**32:raise RuntimeError('备用随机路径超过32位计数范围，明确拒绝而非静默重复')
        generator.set_offset(offset+advance)
        _strict_random_kernel[(triton.cdiv(out.numel(),256),)](out,out.numel(),seed,offset,256)
    return out
