import torch
import triton
import triton.language as tl
_WARPS = 4
_BASE = 1024
_CAP_2B = 8192
_CAP_4B = 4096
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CACHE = {}
_CORES = {}
_RAW_GETTER = None
_RAW_PROBED = False

@triton.jit
def _hsb_stream(go_ptr, x_ptr, out_ptr, N: tl.constexpr, lambd, BLOCK: tl.constexpr):
    """Exact-cover streaming: N is a whole number of (programs x BLOCK) tiles."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, N, step):
        go = tl.load(go_ptr + start + lane)
        xx = tl.load(x_ptr + start + lane)
        tl.store(out_ptr + start + lane, tl.where(tl.abs(xx) <= lambd, tl.zeros_like(go), go))

@triton.jit
def _hsb_stream_tail(go_ptr, x_ptr, out_ptr, N: tl.constexpr, lambd, BLOCK: tl.constexpr):
    """Same streaming hot loop, plus one predicated tile for the ragged tail.

    Coverage: the hot loop fills [0, (N//step)*step); the remaining region is
    shorter than one full stride, so each program covers at most one tail tile,
    and the tail predicate alone guarantees no out-of-range access.
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    full = (N // step) * step
    for start in range(pid * BLOCK, full, step):
        go = tl.load(go_ptr + start + lane)
        xx = tl.load(x_ptr + start + lane)
        tl.store(out_ptr + start + lane, tl.where(tl.abs(xx) <= lambd, tl.zeros_like(go), go))
    off = full + pid * BLOCK + lane
    m = off < N
    go = tl.load(go_ptr + off, m, other=0)
    xx = tl.load(x_ptr + off, m, other=0)
    tl.store(out_ptr + off, tl.where(tl.abs(xx) <= lambd, tl.zeros_like(go), go), m)

@triton.jit
def _hsb_masked(go_ptr, x_ptr, out_ptr, n, lambd, BLOCK: tl.constexpr):
    """Flat one-tile-per-program kernel, kept for the non-contiguous layout path."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    go = tl.load(go_ptr + offs, mask=m, other=0)
    xx = tl.load(x_ptr + offs, mask=m, other=0)
    tl.store(out_ptr + offs, tl.where(tl.abs(xx) <= lambd, tl.zeros_like(go), go), mask=m)

def _resolve_raw():
    """Raw current-stream accessor (~0.35 us) instead of torch.npu.current_stream() (~23 us)."""
    global _RAW_GETTER, _RAW_PROBED
    if _RAW_PROBED:
        return _RAW_GETTER
    _RAW_PROBED = True
    try:
        import torch_npu
        raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
        if raw is not None:
            dev = torch.npu.current_device()
            if int(raw(dev)) == int(torch.npu.current_stream().npu_stream):
                _RAW_GETTER = raw
    except Exception:
        _RAW_GETTER = None
    return _RAW_GETTER

def _stream_getter(dev):
    raw = _resolve_raw()
    if raw is not None:
        return lambda: raw(dev)
    return lambda: torch.npu.current_stream().npu_stream

def _cores_of(dev):
    """Launched concurrency cap: the reported vector-core count for this device."""
    c = _CORES.get(dev)
    if c is not None:
        return c
    c = 0
    try:
        from triton.runtime import driver
        props = driver.active.utils.get_device_properties(dev)
        c = int(props.get('num_vectorcore') or 0)
    except Exception:
        c = 0
    _CORES[dev] = c
    return c

def _uniform_stride(t):
    """Single element stride of a view whose dims collapse into one axis, else None."""
    shape = t.shape
    stride = t.stride()
    dims = []
    for d in range(len(shape) - 1, -1, -1):
        s = shape[d]
        if s == 1:
            continue
        st = stride[d]
        if dims and st == dims[-1][0] * dims[-1][1]:
            dims[-1] = (dims[-1][0] * s, dims[-1][1])
        else:
            dims.append((s, st))
    if not dims:
        return 1
    if len(dims) == 1:
        return dims[0][1]
    return None

def _geometry(n, esize, dev):
    """Task-derived tile, plus a concurrency cap by reported vector cores."""
    cap = _CAP_4B if esize > 2 else _CAP_2B
    block = min(cap, max(32, 1 << (n - 1).bit_length()))
    grid = -(-n // block)
    if grid > 1:
        cores = _cores_of(dev)
        if cores and grid > cores:
            grid = cores
    return block, grid, (n % (grid * block)) != 0

def _build(n, esize, sample, dev):
    """Resolve (compile once) the launcher for this (numel, dtype-size, device)."""
    block, grid, tail = _geometry(n, esize, dev)
    jitfn = _hsb_stream_tail if tail else _hsb_stream
    lam0 = 0.5
    stream_of = _stream_getter(dev)

    def jit_call(g, x, o, lam, _f=jitfn, _b=block, _g=grid):
        _f[(_g,)](g, x, o, n, lam, BLOCK=_b, num_warps=_WARPS)
    try:
        ck = jitfn.warmup(sample, sample, sample, n, lam0, BLOCK=block, num_warps=_WARPS, grid=(1,))
        launcher = ck.run
        md = launcher.metadata
        if getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(md, 'debug_enabled', False) or getattr(md, 'debug', False):
            return jit_call
        L, fn, pm = (launcher.launch, ck.function, ck.packed_metadata)
    except Exception:
        return jit_call

    def call(g, x, o, lam, _L=L, _fn=fn, _pm=pm, _g=grid, _sg=stream_of, _n=n, _b=block, _f=jit_call):
        gp = g.data_ptr()
        xp = x.data_ptr()
        op = o.data_ptr()
        if (gp | xp | op) & 15:
            _f(g, x, o, lam)
            return
        _L(_g, 1, 1, _sg(), _fn, _pm, None, None, None, gp, xp, op, _n, lam, _b)
    return call

def _dispatch(grad_out, self_t, out, n, lambd):
    """Write the hardshrink_backward result of (grad_out, self_t) into ``out``."""
    key = (n, grad_out.dtype, grad_out.get_device())
    call = _CACHE.get(key)
    if call is None:
        call = _build(n, out.element_size(), out, key[2])
        _CACHE[key] = call
    call(grad_out, self_t, out, lambd)

def _layout_fallback(grad_out, self_t, out, n, lambd):
    go = grad_out if grad_out.is_contiguous() else _strict_contiguous(grad_out)
    xx = self_t if self_t.is_contiguous() else _strict_contiguous(self_t)
    _hsb_masked[(n + _BASE - 1) // _BASE,](go, xx, out, n, lambd, BLOCK=_BASE, num_warps=_WARPS)

def hardshrink_backward(grad_out: torch.Tensor, self_t: torch.Tensor, lambd=0.5):
    dt = grad_out.dtype
    if dt not in _DTYPES:
        raise TypeError(f'Unsupported dtype {dt}. Supported dtypes are float16, bfloat16, and float32.')
    if dt is not self_t.dtype:
        raise TypeError(f'Dtype mismatch: grad_out.dtype={dt}, self.dtype={self_t.dtype}')
    if self_t.shape != grad_out.shape:
        raise ValueError(f'Shape mismatch: grad_out.shape={grad_out.shape}, self.shape={self_t.shape}')
    n = grad_out.numel()
    out = _strict_empty_like(grad_out)
    if n == 0:
        return out
    lam = float(lambd)
    if grad_out.is_contiguous() and self_t.is_contiguous():
        _dispatch(grad_out, self_t, out, n, lam)
    else:
        _layout_fallback(grad_out, self_t, out, n, lam)
    return out

def hardshrink_backward_grad_input(grad_out: torch.Tensor, self_t: torch.Tensor, lambd=0.5, *, grad_input=None):
    if grad_input is None:
        raise ValueError("Argument 'grad_input' must be provided for hardshrink_backward.grad_input.")
    if self_t.shape != grad_out.shape or grad_input.shape != grad_out.shape:
        raise ValueError(f'Shape mismatch: grad_out.shape={grad_out.shape}, self.shape={self_t.shape}, grad_input.shape={grad_input.shape}')
    _strict_copy_into(grad_input, hardshrink_backward(grad_out, self_t, lambd))
    return grad_input
run = hardshrink_backward

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
