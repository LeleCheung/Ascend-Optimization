"""FlagGems hardswish math with bounded streaming tiles and compiled launches."""
import torch
import torch_npu
import triton
import triton.language as tl
_ENTRIES = {}
_STREAM = {}

@triton.jit
def _out_hardswish_stream(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        offset = start + lane
        x = tl.load(X + offset, offset < N, other=0).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        result = x * (inner * (1.0 / 6.0))
        tl.store(Y + offset, result, offset < N)

@triton.jit
def _out_hardswish_pair(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * (2 * BLOCK)
    for start in range(tl.program_id(0) * (2 * BLOCK), N, step):
        first = start + lane
        second = first + BLOCK
        a = tl.load(X + first, first < N, other=0).to(tl.float32)
        b = tl.load(X + second, second < N, other=0).to(tl.float32)
        ia = tl.minimum(tl.maximum(a + 3.0, 0.0), 6.0)
        ib = tl.minimum(tl.maximum(b + 3.0, 0.0), 6.0)
        ya = a * (ia * (1.0 / 6.0))
        yb = b * (ib * (1.0 / 6.0))
        tl.store(Y + first, ya, first < N)
        tl.store(Y + second, yb, second < N)

def _out_stream_getter(device):
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

def _out_entry(x, out, n, block, device):
    entry = None
    kernel = _out_hardswish_pair if n >= 2621440 else _out_hardswish_stream
    try:
        compiled = kernel.warmup(x, out, n, block, grid=(1,))
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False) or getattr(launcher.metadata, 'debug', False)):
            entry = (launcher.launch, compiled.function, compiled.packed_metadata, _out_stream_getter(device))
    except Exception:
        entry = None
    return entry

def _out_compute(x, out):
    n = x.numel()
    if not n:
        return out
    block = min(4096, max(32, 1 << (n - 1).bit_length()))
    grid = min(40, (n + block - 1) // block) if n >= 131072 else (n + block - 1) // block
    kernel = _out_hardswish_pair if n >= 2621440 else _out_hardswish_stream
    xp, yp = (x.data_ptr(), out.data_ptr())
    if (xp | yp) & 15:
        kernel[grid,](x, out, n, block)
        return out
    device = x.device.index
    key = (x.dtype, out.dtype, device, n, block)
    if key not in _ENTRIES:
        _ENTRIES[key] = _out_entry(x, out, n, block, device)
    entry = _ENTRIES[key]
    if entry is None:
        kernel[grid,](x, out, n, block)
    else:
        entry[0](grid, 1, 1, entry[3](), entry[1], entry[2], None, None, None, xp, yp, n, block)
    return out

def _out_fallback(x, out):
    if out.shape != x.shape:
        out.resize_(x.shape)
    contiguous = _strict_contiguous(x)
    if out.is_contiguous():
        _out_compute(contiguous, out)
    else:
        temporary = torch.empty(contiguous.shape, dtype=contiguous.dtype, device=contiguous.device)
        _out_compute(contiguous, temporary)
        _strict_copy_into(out, temporary)
    return out

def run(x, out):
    return _out_fallback(x, out)

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
