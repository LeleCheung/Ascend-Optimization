"""支持输入/out非连续布局的偶数舍入；使用FP32计算，保留特殊值和零符号。"""
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _round_value(x, DECIMALS: tl.constexpr):
    xf = x.to(tl.float32)
    scale: tl.constexpr = 10.0 ** DECIMALS
    value = xf * scale
    magnitude = tl.abs(value)
    shifted = magnitude + 8388608.0
    rounded = shifted - 8388608.0
    rounded = tl.where(magnitude >= 8388608.0, magnitude, rounded)
    sign = value.to(tl.uint32, bitcast=True) & 2147483648
    signed = (rounded.to(tl.uint32, bitcast=True) & 2147483647 | sign).to(tl.float32, bitcast=True)
    if DECIMALS != 0:
        signed = tl.div_rn(signed, scale)
    return tl.where((xf != xf) | (tl.abs(xf) == float('inf')), xf, signed)

@triton.jit
def _round_flat(X, Y, N, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    tl.store(Y + index, _round_value(x, DECIMALS).to(Y.dtype.element_ty), index < N)

@triton.jit
def _round_strided(X, Y, N, SHAPE: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = index
    xi = tl.full((BLOCK,), 0, tl.int64)
    yi = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = remainder % SHAPE[axis]
        remainder = remainder // SHAPE[axis]
        xi += coordinate.to(tl.int64) * XS[axis]
        yi += coordinate.to(tl.int64) * YS[axis]
    x = tl.load(X + xi, index < N, other=0)
    tl.store(Y + yi, _round_value(x, DECIMALS).to(Y.dtype.element_ty), index < N)
_ENTRIES = {}
_STREAM = {}

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

def _dispatch(x, out, decimals):
    dtype = x.dtype
    device_object = x.device
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('当前实现支持float16、bfloat16、float32')
    if out is None:
        out = _strict_empty_like(x, memory_format=torch.contiguous_format)
    if out.shape != x.shape or out.dtype != dtype or out.device != device_object:
        raise ValueError('out必须与输入具有相同形状、dtype和设备')
    n = x.numel()
    if not n:
        return out
    if not x.is_contiguous() or not out.is_contiguous():
        _round_strided[(n + 255) // 256,](x, out, n, tuple(x.shape), tuple(x.stride()), tuple(out.stride()), decimals, BLOCK=256)
        return out
    device = device_object.index
    key = (device, dtype, n, decimals)
    entry = _ENTRIES.get(key)
    if entry is None:
        block = min(4096, max(256, 1 << ((n + 7) // 8 - 1).bit_length()))
        grid = (n + block - 1) // block
        launch = None
        try:
            compiled = _round_flat.warmup(x, out, n, decimals, BLOCK=block, grid=(1,))
            launcher = compiled.run
            if not (getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False) or getattr(launcher.metadata, 'debug', False)):
                launch = (launcher.launch, compiled.function, compiled.packed_metadata)
        except Exception:
            pass
        entry = (grid, block, launch, _stream_getter(device) if launch else None)
        _ENTRIES[key] = entry
    launch = entry[2]
    xp, yp = (x.data_ptr(), out.data_ptr())
    if launch and (not (xp | yp) & 15):
        launch[0](entry[0], 1, 1, entry[3](), launch[1], launch[2], None, None, None, xp, yp, n, decimals, entry[1])
    else:
        _round_flat[entry[0],](x, out, n, decimals, BLOCK=entry[1])
    return out

def run(input, *, decimals=0, out=None):
    return _dispatch(input, out, decimals)

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
