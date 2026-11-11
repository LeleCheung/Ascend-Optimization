"""Self-contained Triton implementation of ``aten._conj_physical``.

Semantics (verified against the reference on the target):
  * the result is always a freshly allocated tensor -- for real dtypes the
    conjugate is the identity, but ``_conj_physical`` still returns a copy, so
    the input must never be aliased or returned;
  * the input may carry the lazy conjugate bit; it stores ``conj(logical)`` and
    the operator conjugates the *logical* value, hence
      - bit clear -> output = conj(storage)   (imaginary part negated)
      - bit set   -> output = storage          (imaginary part unchanged)
    The bit becomes a constexpr ``NEG`` and is cleared with a metadata-only
    ``input.conj()`` view, so the kernel reads plain storage;
  * the reference (whose only backend here is the CPU fallback) materialises a
    contiguous output for every layout, so the result is contiguous and the
    input is read through its real strides.

Kernels: a flat vector kernel over contiguous storage (all timing inputs) and a
rank-generic strided kernel for arbitrary layouts.

Dense contiguous calls take a pre-bound launch path: the first call for a
(kernel, dtype, tile, pointer-alignment) specialization compiles through the
ordinary ``kernel[grid](...)`` call; later calls reuse that compiled kernel and
invoke its generated launcher directly, passing this call's pointers, element
count and *current* stream. Nothing about a tensor (data, shape, stream, dtype)
is cached -- only the compiled artifact.
"""
import torch
import triton
import triton.language as tl
try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _raw_stream
except Exception:
    _raw_stream = None
_TINY_MAX = 4096
_MID_MAX = 262144
_COPY_TINY, _COPY_MID, _COPY_LARGE = (1024, 8192, 16384)
_CONJ_TINY, _CONJ_MID, _CONJ_LARGE = (1024, 4096, 8192)
_STR_TINY, _STR_MID, _STR_LARGE = (1024, 2048, 4096)
_STR_COMPLEX_BLOCK = 1024
_copy_cache = {}
_conj_cache = {}
_device = None
_stream_fn = None

@triton.jit(do_not_specialize=['n'])
def _copy_flat_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    """Flat element copy: real dtypes need no arithmetic at all."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)

@triton.jit(do_not_specialize=['n'])
def _conj_flat_kernel(src_ptr, dst_ptr, n, NEG: tl.constexpr, BLOCK: tl.constexpr):
    """Flat copy over the interleaved real/imaginary float stream.

    ``n`` counts scalars, so even offsets hold real parts and odd offsets hold
    imaginary parts. Negating only the odd lanes keeps both the load and the
    store fully contiguous, and a sign flip is exact for every float value.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src_ptr + offs, mask=mask)
    if NEG:
        v = tl.where(offs & 1 == 1, -v, v)
    tl.store(dst_ptr + offs, v, mask=mask)

@triton.jit
def _conj_strided_kernel(src_ptr, dst_ptr, n, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4, t5, t6, t7, NEG: tl.constexpr, IS_COMPLEX: tl.constexpr, NDIM: tl.constexpr, BLOCK: tl.constexpr):
    """Rank-generic strided read into a contiguous write.

    ``s*`` are the logical sizes (last axis fastest), ``t*`` the input strides
    in scalar units. The destination is contiguous, so its offset is a plain
    multiple of the logical linear index.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = tl.zeros_like(idx)
    if NDIM >= 1:
        off += rem % s7 * t7
        rem = rem // s7
    if NDIM >= 2:
        off += rem % s6 * t6
        rem = rem // s6
    if NDIM >= 3:
        off += rem % s5 * t5
        rem = rem // s5
    if NDIM >= 4:
        off += rem % s4 * t4
        rem = rem // s4
    if NDIM >= 5:
        off += rem % s3 * t3
        rem = rem // s3
    if NDIM >= 6:
        off += rem % s2 * t2
        rem = rem // s2
    if NDIM >= 7:
        off += rem % s1 * t1
        rem = rem // s1
    if NDIM >= 8:
        off += rem % s0 * t0
    if IS_COMPLEX:
        out_off = idx * 2
        re = tl.load(src_ptr + off, mask=mask)
        im = tl.load(src_ptr + off + 1, mask=mask)
        if NEG:
            im = -im
        tl.store(dst_ptr + out_off, re, mask=mask)
        tl.store(dst_ptr + out_off + 1, im, mask=mask)
    else:
        tl.store(dst_ptr + idx, tl.load(src_ptr + off, mask=mask), mask=mask)

def _resolve_stream(device):
    global _device, _stream_fn
    _device = device
    if _raw_stream is not None:
        _stream_fn = lambda: _raw_stream(device)
    else:
        _stream_fn = lambda: triton.runtime.driver.active.get_current_stream(device)

def _cur_stream():
    device = torch.npu.current_device()
    if device != _device:
        _resolve_stream(device)
    return _stream_fn()

def _bind_copy(src, dst, n, block, grid):
    """Compile and remember the launcher for this dense-copy specialization."""
    try:
        if _stream_fn is None:
            _resolve_stream(torch.npu.current_device())
        compiled = _copy_flat_kernel.warmup(src, dst, n, BLOCK=block, grid=(grid, 1, 1))
        _copy_cache[src.dtype, block, src.data_ptr() & 15, dst.data_ptr() & 15] = (compiled.run.launch, compiled.function, compiled.packed_metadata)
    except Exception:
        pass

def _bind_conj(src, dst, n, neg, block, grid):
    """Compile and remember the launcher for this dense-conjugate specialization.

    The specialization is keyed on the interleaved *real* views (the kernel reads
    a float stream), so warmup receives the same view the ordinary launch passes;
    their base pointer is the complex tensor's pointer, so the cached launcher
    matches the raw pointers used later.
    """
    try:
        if _stream_fn is None:
            _resolve_stream(torch.npu.current_device())
        compiled = _conj_flat_kernel.warmup(torch.view_as_real(src), torch.view_as_real(dst), n, NEG=neg, BLOCK=block, grid=(grid, 1, 1))
        _conj_cache[src.dtype, block, neg, src.data_ptr() & 15, dst.data_ptr() & 15] = (compiled.run.launch, compiled.function, compiled.packed_metadata)
    except Exception:
        pass

def _collapse(shape, strides):
    """Merge adjacent axes whose strides compose exactly (pure index rewrite)."""
    shape = list(shape)
    strides = list(strides)
    i = len(shape) - 1
    while i > 0:
        if shape[i] != 0 and strides[i - 1] == strides[i] * shape[i]:
            shape[i - 1] *= shape[i]
            strides[i - 1] = strides[i]
            del shape[i]
            del strides[i]
        i -= 1
    return (shape, strides)

def _launch_strided(src, out, n, neg, is_complex, block):
    shape, strides = _collapse(src.shape, src.stride())
    if is_complex:
        strides = [s * 2 for s in strides]
    if len(shape) > 8:
        step = n // shape[0]
        for k in range(shape[0]):
            _launch_strided(src[k], out[k], step, neg, is_complex, block)
        return
    rank = len(shape)
    pad_s = [1] * 8
    pad_t = [0] * 8
    for d in range(rank):
        pad_s[8 - rank + d] = shape[d]
        pad_t[8 - rank + d] = strides[d]
    carrier = torch.view_as_real(src) if is_complex else src
    grid = (triton.cdiv(n, block),)
    _conj_strided_kernel[grid](carrier, torch.view_as_real(out) if is_complex else out, n, pad_s[0], pad_s[1], pad_s[2], pad_s[3], pad_s[4], pad_s[5], pad_s[6], pad_s[7], pad_t[0], pad_t[1], pad_t[2], pad_t[3], pad_t[4], pad_t[5], pad_t[6], pad_t[7], NEG=neg, IS_COMPLEX=is_complex, NDIM=rank, BLOCK=block)

def _block_for(n, small, mid, large):
    if n <= _TINY_MAX:
        return small
    if n <= _MID_MAX:
        return mid
    return large

def run(input):
    n = input.numel()
    has_bit = input.is_conj()
    contig = input.is_contiguous()
    if contig and (not has_bit):
        out = _strict_empty_like(input)
    else:
        out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if n == 0:
        return out
    if input.is_complex():
        src = input.conj() if has_bit else input
        neg = not has_bit
        if contig:
            nf = n + n
            block = _block_for(nf, _CONJ_TINY, _CONJ_MID, _CONJ_LARGE)
            grid = -(-nf // block)
            sp = src.data_ptr()
            dp = out.data_ptr()
            entry = _conj_cache.get((input.dtype, block, neg, sp & 15, dp & 15))
            if entry is None:
                _conj_flat_kernel[grid,](torch.view_as_real(src), torch.view_as_real(out), nf, NEG=neg, BLOCK=block)
                _bind_conj(src, out, nf, neg, block, grid)
            else:
                entry[0](grid, 1, 1, _cur_stream(), entry[1], entry[2], None, None, None, sp, dp, nf, neg, block)
            return out
        _launch_strided(src, out, n, neg, True, _STR_COMPLEX_BLOCK)
        return out
    if contig:
        block = _block_for(n, _COPY_TINY, _COPY_MID, _COPY_LARGE)
        grid = -(-n // block)
        sp = input.data_ptr()
        dp = out.data_ptr()
        entry = _copy_cache.get((input.dtype, block, sp & 15, dp & 15))
        if entry is None:
            _copy_flat_kernel[grid,](input, out, n, BLOCK=block)
            _bind_copy(input, out, n, block, grid)
        else:
            entry[0](grid, 1, 1, _cur_stream(), entry[1], entry[2], None, None, None, sp, dp, n, block)
        return out
    _launch_strided(input, out, n, False, False, _block_for(n, _STR_TINY, _STR_MID, _STR_LARGE))
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
