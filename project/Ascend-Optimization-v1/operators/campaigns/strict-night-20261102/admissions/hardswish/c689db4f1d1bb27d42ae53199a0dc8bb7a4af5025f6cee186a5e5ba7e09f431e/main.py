"""FlagGems hardswish: streaming tiles with a minimal per-call host path.

The device kernel and the block/grid policy are unchanged from the seed: one
masked streaming launch whose stride is the launched program count.  Only the
host path changed.  Per call we now do a contiguity call, one allocation, one
integer compare and an integer-address launch of the cached compiled kernel;
the Triton binder/cache-key work that used to dominate tiny workloads is gone.
Any unaligned pointer falls back to the bound JIT call so the compiler's
alignment assumptions can never be violated, and the launch-concurrency cap is
the vector-core count reported by the target instead of a device literal.
"""
import torch
import torch_npu
import triton
import triton.language as tl
_STREAM = {}
_PLANS = {}
_FAST = [-1, None, -1, None]

@triton.jit
def _hardswish_stream(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        offset = start + lane
        x = tl.load(X + offset, offset < N, other=0).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        result = x * inner * (1.0 / 6.0)
        tl.store(Y + offset, result, offset < N)

def _launch_jit(y, out, grid, n, block):
    """Bound JIT launch, used whenever the compiled fast path cannot be used."""
    _hardswish_stream[grid,](y, out, n, block)

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

def _geometry(n, device, dtype):
    if n < 131072:
        block = min(4096, max(32, 1 << (n - 1).bit_length()))
        return (block, (n + block - 1) // block)
    # Large regime: bigger tiles amortise the exposed per-iteration latency of
    # the dynamic streaming loop; 16-bit tiles carry twice the elements inside
    # the same on-chip footprint.
    block = 16384 if dtype.itemsize <= 2 else 8192
    grid = (n + block - 1) // block
    cores = _vector_cores(device)
    if cores and grid > cores:
        grid = cores
    return (block, grid)

def _build(y, out, n, device):
    """Compile this specialization once and return a ready-to-call closure."""
    block, grid = _geometry(n, device, y.dtype)
    grid3 = (grid, 1, 1)
    jit = _hardswish_stream
    launch = fn = packed = stream = None
    try:
        compiled = jit.warmup(y, out, n, block, grid=(1,))
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False) or getattr(launcher.metadata, 'debug', False)):
            launch = launcher.launch
            fn = compiled.function
            packed = compiled.packed_metadata
            stream = _stream_getter(device)
    except Exception:
        launch = None
    if launch is None:

        def call(y, out, _g=grid, _n=n, _b=block):
            _launch_jit(y, out, _g, _n, _b)
        return call

    def call(y, out, _l=launch, _f=fn, _p=packed, _g=grid3, _r=stream, _n=n, _b=block, _grid=grid):
        xp = y.data_ptr()
        yp = out.data_ptr()
        if (xp | yp) & 15:
            _launch_jit(y, out, _grid, _n, _b)
        else:
            _l(_g[0], 1, 1, _r(), _f, _p, None, None, None, xp, yp, _n, _b)
    return call

def _slow(y, out):
    n = y.numel()
    if not n:
        return
    device = y.get_device()
    key = (device, y.dtype, n)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _build(y, out, n, device)
    _FAST[0] = device
    _FAST[1] = y.dtype
    _FAST[2] = n
    _FAST[3] = plan
    plan(y, out)

def run(x):
    y = _strict_contiguous(x)
    out = _strict_empty_like(y)
    n = y.numel()
    if n and y.dtype is _FAST[1] and (n == _FAST[2]) and (y.get_device() == _FAST[0]):
        _FAST[3](y, out)
    else:
        _slow(y, out)
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
