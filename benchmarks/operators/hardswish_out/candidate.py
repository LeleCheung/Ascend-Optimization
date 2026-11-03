"""严格Triton hardswish_out。复用本批已核验hardswish的完整块流式设备核，
连续输出直接写入调用方out，避免创建中间结果；真实stride分支由Triton复制。
保留NaN/Inf、负零及当前stream；缓存区分输入/输出dtype、设备与大小。
"""
import torch
import torch_npu
import triton
import triton.language as tl
_STREAM = {}
_PLANS = {}
_EMPTY_LIKE = torch.empty_like
# dtype, n, device, launch, grid, block, opts, fn, packed, getter, has_launch
_FAST = [None, -1, -1, None, 1, 4096, {}, None, None, None, False, None]

@triton.jit
def _hardswish_stream(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    # Hot loop: every tile here is fully in range, so the transfers stay
    # unmasked contiguous bursts.  Only the trailing region needs a predicate.
    full = (N // step) * step
    for start in range(pid * BLOCK, full, step):
        x = tl.load(X + start + lane).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        tl.store(Y + start + lane, x * inner * (1.0 / 6.0))
    offset = full + pid * BLOCK + lane
    tail = offset < N
    x = tl.load(X + offset, tail, other=0).to(tl.float32)
    inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
    tl.store(Y + offset, x * inner * (1.0 / 6.0), tail)

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
    launch = fn = packed = stream = None
    try:
        compiled = _hardswish_stream.warmup(y, out, n, block, grid=(1,))
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False) or getattr(launcher.metadata, 'debug', False)):
            launch = launcher.launch
            fn = compiled.function
            packed = compiled.packed_metadata
            stream = _stream_getter(device)
    except Exception:
        launch = None
    return [y.dtype, n, device, launch, grid, block, {}, fn, packed, stream,
            launch is not None, out.dtype]

def _slow(y, out, n):
    if not n:
        return
    device = y.get_device()
    key = (device, y.dtype, out.dtype, n)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _build(y, out, n, device)
    _FAST[:] = plan
    plan[0], plan[1], plan[2] = y.dtype, n, device
    xp = y.data_ptr()
    yp = out.data_ptr()
    if plan[10] and not ((xp | yp) & 15):
        plan[3](plan[4], 1, 1, plan[9](), plan[7], plan[8], None, None, None,
                xp, yp, n, plan[5])
    elif n:
        _hardswish_stream[plan[4],](y, out, n, plan[5])

def run(x, out):
    # out仅调整元数据/存储，不执行宿主数值或复制。
    if out.shape != x.shape:
        out.resize_(x.shape)
    device = x.get_device()
    if out.get_device() != device:
        raise ValueError('输入和out必须位于同一设备')
    if not x.is_contiguous():
        x = _strict_clone(x)
    n = x.numel()
    if not out.is_contiguous():
        temporary = torch.empty_like(x, dtype=out.dtype)
        _slow(x, temporary, n)
        _strict_copy_into(out, temporary)
        return out
    f = _FAST
    if x.dtype is f[0] and out.dtype is f[11] and n == f[1] and device == f[2]:
        xp = x.data_ptr()
        yp = out.data_ptr()
        if f[10] and not ((xp | yp) & 15):
            f[3](f[4], 1, 1, f[9](), f[7], f[8], None, None, None,
                 xp, yp, n, f[5])
        elif n:
            _hardswish_stream[f[4],](x, out, n, f[5])
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


# 宿主桥只处理元数据与编译参数；所有计算及非连续回退仍为上述Triton。
import hashlib as _host_hash
import importlib.util as _host_import
import sys as _host_sys
from pathlib import Path as _HostPath
_host_binary = _HostPath("/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-night-20261102/host-glue/out-v1/_hgkda_host_bridge_out_v1.so")
assert _host_hash.sha256(_host_binary.read_bytes()).hexdigest() == "11e4286a9b31280a70ed93fa3c606de41caddd18e54b4fdada4b6dff51f4349e"
assert torch.__version__ == "2.9.0+cpu" and torch._C._GLIBCXX_USE_CXX11_ABI == True
_bridge_module = _host_sys.modules.get("_hgkda_host_bridge_out_v1")
if _bridge_module is None:
    _host_spec = _host_import.spec_from_file_location("_hgkda_host_bridge_out_v1", _host_binary)
    _bridge_module = _host_import.module_from_spec(_host_spec)
    _host_spec.loader.exec_module(_bridge_module)
    _host_sys.modules["_hgkda_host_bridge_out_v1"] = _bridge_module
else:
    assert _HostPath(_bridge_module.__file__).resolve() == _host_binary.resolve()
_python_triton_run = run


def _bridge_fallback(x, out):
    result = _python_triton_run(x, out)
    n = x.numel()
    if (n and x.is_contiguous() and out.is_contiguous()
            and x.shape == out.shape and x.device == out.device
            and not ((x.data_ptr() | out.data_ptr()) & 15)):
        plan = _PLANS.get((x.get_device(), x.dtype, out.dtype, n))
        if plan is not None and plan[10]:
            arguments = (plan[4], 1, 1, None, plan[7], plan[8], None, None, None,
                         None, None, n, plan[5])
            _BRIDGE.register_plan(x, out, plan[3], arguments, plan[9])
    return result


_BRIDGE = _bridge_module.OutLaunchCache(_bridge_fallback)


def run(x, out):
    return _BRIDGE.run(x, out)
