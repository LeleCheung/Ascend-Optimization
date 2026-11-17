import torch
import triton
import triton.language as tl
_BIG = tl.constexpr(1 << 24)

@triton.jit
def _combine_max_idx(a_val, a_idx, b_val, b_idx):
    take_b = (b_val > a_val) | (b_val != b_val)
    return (tl.where(take_b, b_val, a_val), tl.where(take_b, b_idx, a_idx))

@triton.jit
def _round_coordinate(value, DT: tl.constexpr):
    if DT == tl.float16:
        a = tl.abs(value)
        bits = a.to(tl.int32, bitcast=True)
        rounded = (bits + 4095 + (bits >> 13 & 1) & -8192).to(tl.float32, bitcast=True)
        units = tl.minimum(a, 6.103515625e-05) * 16777216.0
        lower = tl.floor(units)
        fraction = units - lower
        add = (fraction > 0.5) | (fraction == 0.5) & (lower.to(tl.int32) & 1 != 0)
        subnormal = (lower + add.to(tl.float32)) * 5.960464477539063e-08
        rounded = tl.where(a < 6.103515625e-05, subnormal, rounded)
        rounded = tl.where(rounded > 65504.0, float('inf'), rounded)
        return tl.where(value < 0.0, -rounded, rounded)
    elif DT == tl.bfloat16:
        bits = value.to(tl.int32, bitcast=True)
        rounded = (bits + 32767 + (bits >> 16 & 1) & -65536).to(tl.float32, bitcast=True)
        return tl.where(tl.abs(value) < float('inf'), rounded, value)
    else:
        return value

@triton.jit
def _fractional_max_pool3d_general_kernel(in_ptr, out_ptr, idx_ptr, samp_ptr, s_n, s_c, s_d, s_h, s_w, ss_n, ss_c, in_c, in_d, in_h, in_w, out_d, out_h, out_w, alpha_d, alpha_h, alpha_w, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, WRITE_IDX: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_hw_blocks = tl.cdiv(out_h, BLOCK_H) * num_w_blocks
    d_idx = pid_dhw // num_hw_blocks
    hw = pid_dhw % num_hw_blocks
    h_block = hw // num_w_blocks
    w_block = hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c
    ho = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    wo = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = ho < out_h
    w_mask = wo < out_w
    neg_inf = float('-inf')
    base = in_ptr + n_idx * s_n + c_idx * s_c
    samp = samp_ptr + n_idx * ss_n + c_idx * ss_c
    sample_d = tl.load(samp).to(tl.float32)
    sample_h = tl.load(samp + 1).to(tl.float32)
    sample_w = tl.load(samp + 2).to(tl.float32)
    scalar_dtype: tl.constexpr = samp_ptr.dtype.element_ty
    alpha_d = _round_coordinate(_round_coordinate(tl.cast(in_d - KD, tl.float32), scalar_dtype) / _round_coordinate(tl.cast(tl.maximum(out_d - 1, 1), tl.float32), scalar_dtype), scalar_dtype)
    alpha_h = _round_coordinate(_round_coordinate(tl.cast(in_h - KH, tl.float32), scalar_dtype) / _round_coordinate(tl.cast(tl.maximum(out_h - 1, 1), tl.float32), scalar_dtype), scalar_dtype)
    alpha_w = _round_coordinate(_round_coordinate(tl.cast(in_w - KW, tl.float32), scalar_dtype) / _round_coordinate(tl.cast(tl.maximum(out_w - 1, 1), tl.float32), scalar_dtype), scalar_dtype)
    base_h = _round_coordinate(sample_h * alpha_h, scalar_dtype).to(tl.int32)
    base_w = _round_coordinate(sample_w * alpha_w, scalar_dtype).to(tl.int32)
    base_d = _round_coordinate(sample_d * alpha_d, scalar_dtype).to(tl.int32)
    shifted_h = _round_coordinate(ho.to(tl.float32) + sample_h, scalar_dtype)
    shifted_w = _round_coordinate(wo.to(tl.float32) + sample_w, scalar_dtype)
    shifted_d = _round_coordinate(d_idx.to(tl.float32) + sample_d, scalar_dtype)
    h_start = _round_coordinate(shifted_h * alpha_h, scalar_dtype).to(tl.int32) - base_h
    w_start = _round_coordinate(shifted_w * alpha_w, scalar_dtype).to(tl.int32) - base_w
    d_start = _round_coordinate(shifted_d * alpha_d, scalar_dtype).to(tl.int32) - base_d
    h_start = tl.where(ho == out_h - 1, in_h - KH, h_start)
    w_start = tl.where(wo == out_w - 1, in_w - KW, w_start)
    d_start = tl.where(d_idx == out_d - 1, in_d - KD, d_start)
    h_base = h_start[:, None]
    w_base = w_start[None, :]
    best_i = d_start * in_h * in_w + h_base * in_w + w_base
    best_v = tl.full((BLOCK_H, BLOCK_W), neg_inf, dtype=in_ptr.type.element_ty)
    for kd in tl.static_range(KD):
        d_in = d_start + kd
        for kh in tl.static_range(KH):
            h_in = h_base + kh
            for kw in tl.static_range(KW):
                w_in = w_base + kw
                m = h_mask[:, None] & w_mask[None, :] & (d_in >= 0) & (d_in < in_d) & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
                val = tl.load(base + d_in * s_d + h_in * s_h + w_in * s_w, mask=m, other=neg_inf)
                idx = d_in * in_h * in_w + h_in * in_w + w_in
                best_v, best_i = _combine_max_idx(best_v, best_i, val, idx)
    out_off = ((pid_nc * out_d + d_idx) * out_h + ho[:, None]) * out_w + wo[None, :]
    store_mask = h_mask[:, None] & w_mask[None, :]
    tl.store(out_ptr + out_off, best_v, mask=store_mask)
    if WRITE_IDX:
        tl.store(idx_ptr + out_off, best_i.to(tl.int64), mask=store_mask)

def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 1:
        return (value[0], value[0], value[0])
    return (value[0], value[1], value[2])

def fractional_max_pool3d(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices: bool=True, _random_samples=None):
    """Fractional max pooling 3d (stochastic pooling regions).

    Args:
        input: 5D (N, C, D, H, W) or 4D (C, D, H, W) input tensor.
        kernel_size: Pooling window size (int or triple).
        output_size: Target output spatial size (int or triple).
        output_ratio: Alternative to output_size, output as a ratio.
        return_indices: Whether to also return the max indices.
        _random_samples: Optional tensor of shape (N, C, 3) (or (C, 3)) with
            values in [0, 1) selecting the pooling region offsets.
    """
    squeeze = False
    if input.dim() == 4:
        input = input.unsqueeze(0)
        squeeze = True
        if _random_samples is not None and _random_samples.dim() == 2:
            _random_samples = _random_samples.unsqueeze(0)
    elif input.dim() != 5:
        raise AssertionError(f'Expected 5D input, got {input.dim()}D')
    in_n, in_c, in_d, in_h, in_w = input.shape
    kernel_d, kernel_h, kernel_w = _triple(kernel_size)
    if output_size is not None:
        out_d, out_h, out_w = _triple(output_size)
    elif output_ratio is not None:
        if isinstance(output_ratio, float):
            ratio_d = ratio_h = ratio_w = output_ratio
        else:
            ratio_d, ratio_h, ratio_w = output_ratio
        out_d = int(in_d * ratio_d)
        out_h = int(in_h * ratio_h)
        out_w = int(in_w * ratio_w)
    else:
        raise ValueError('Either output_size or output_ratio must be specified')
    out_d = int(out_d)
    out_h = int(out_h)
    out_w = int(out_w)
    assert out_d + kernel_d - 1 <= in_d, f'fractional_max_pool3d: kernel_d ({kernel_d}) + output_d ({out_d}) - 1 exceeds input_d ({in_d})'
    assert out_h + kernel_h - 1 <= in_h, f'fractional_max_pool3d: kernel_h ({kernel_h}) + output_h ({out_h}) - 1 exceeds input_h ({in_h})'
    assert out_w + kernel_w - 1 <= in_w, f'fractional_max_pool3d: kernel_w ({kernel_w}) + output_w ({out_w}) - 1 exceeds input_w ({in_w})'
    out_shape = (in_n, in_c, out_d, out_h, out_w)
    output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    if return_indices:
        indices = torch.empty(out_shape, dtype=torch.int64, device=input.device)
    else:
        indices = output
    if output.numel() == 0:
        if squeeze:
            output = output.squeeze(0)
            if return_indices:
                indices = indices.squeeze(0)
        return (output, indices) if return_indices else output
    alpha_d = (in_d - kernel_d) / (out_d - 1) if out_d > 1 else 0.0
    alpha_h = (in_h - kernel_h) / (out_h - 1) if out_h > 1 else 0.0
    alpha_w = (in_w - kernel_w) / (out_w - 1) if out_w > 1 else 0.0
    s_n, s_c, s_d, s_h, s_w = input.stride()
    if _random_samples is None:
        _random_samples = _strict_rand((in_n, in_c, 3), device=input.device, dtype=input.dtype)
    else:
        assert _random_samples.shape == (in_n, in_c, 3), f'_random_samples must have shape ({in_n}, {in_c}, 3), got {_random_samples.shape}'
        if _random_samples.stride(-1) != 1:
            _random_samples = _strict_contiguous(_random_samples)
    block_h = 1
    block_w = 128
    grid = (in_n * in_c, triton.cdiv(out_d * out_h * out_w, block_w))
    _fractional_max_pool3d_general_kernel[grid](input, output, indices, _random_samples, s_n, s_c, s_d, s_h, s_w, _random_samples.stride(0), _random_samples.stride(1), in_c, in_d, in_h, in_w, out_d, out_h, out_w, alpha_d, alpha_h, alpha_w, kernel_d, kernel_h, kernel_w, block_h, block_w, return_indices)
    if squeeze:
        output = output.squeeze(0)
        if return_indices:
            indices = indices.squeeze(0)
    return (output, indices) if return_indices else output
run = fractional_max_pool3d

@triton.jit
def _pool_flat_kernel(in_ptr, out_ptr, idx_ptr, samp_ptr, s_n, s_c, s_d, s_h, s_w, ss_n, ss_c, in_c, in_d, in_h, in_w, out_d, out_h, out_w, alpha_d, alpha_h, alpha_w, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, WRITE_IDX: tl.constexpr):
    nc = tl.program_id(0)
    linear = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = linear < out_d * out_h * out_w
    wo = linear % out_w
    ho = linear // out_w % out_h
    do = linear // (out_h * out_w)
    ni = nc // in_c
    ci = nc % in_c
    samp = samp_ptr + ni * ss_n + ci * ss_c
    sd = tl.load(samp).to(tl.float32)
    sh = tl.load(samp + 1).to(tl.float32)
    sw = tl.load(samp + 2).to(tl.float32)
    dt: tl.constexpr = samp_ptr.dtype.element_ty
    ad = _round_coordinate(_round_coordinate(tl.cast(in_d - KD, tl.float32), dt) / _round_coordinate(tl.cast(tl.maximum(out_d - 1, 1), tl.float32), dt), dt)
    ah = _round_coordinate(_round_coordinate(tl.cast(in_h - KH, tl.float32), dt) / _round_coordinate(tl.cast(tl.maximum(out_h - 1, 1), tl.float32), dt), dt)
    aw = _round_coordinate(_round_coordinate(tl.cast(in_w - KW, tl.float32), dt) / _round_coordinate(tl.cast(tl.maximum(out_w - 1, 1), tl.float32), dt), dt)
    ds = _round_coordinate(_round_coordinate(do.to(tl.float32) + sd, dt) * ad, dt).to(tl.int32) - _round_coordinate(sd * ad, dt).to(tl.int32)
    hs = _round_coordinate(_round_coordinate(ho.to(tl.float32) + sh, dt) * ah, dt).to(tl.int32) - _round_coordinate(sh * ah, dt).to(tl.int32)
    ws = _round_coordinate(_round_coordinate(wo.to(tl.float32) + sw, dt) * aw, dt).to(tl.int32) - _round_coordinate(sw * aw, dt).to(tl.int32)
    ds = tl.where(do == out_d - 1, in_d - KD, ds)
    hs = tl.where(ho == out_h - 1, in_h - KH, hs)
    ws = tl.where(wo == out_w - 1, in_w - KW, ws)
    best = tl.full((BLOCK_W,), float('-inf'), tl.float32)
    index = ds * in_h * in_w + hs * in_w + ws
    base = in_ptr + ni * s_n + ci * s_c
    for di in tl.static_range(KD):
        for hi in tl.static_range(KH):
            for wi in tl.static_range(KW):
                d = ds + di
                h = hs + hi
                w = ws + wi
                valid = mask & (d >= 0) & (d < in_d) & (h >= 0) & (h < in_h) & (w >= 0) & (w < in_w)
                value = tl.load(base + d * s_d + h * s_h + w * s_w, mask=valid, other=float('-inf')).to(tl.float32)
                take = (value > best) | (value != value)
                best = tl.where(take, value, best)
                index = tl.where(take, d * in_h * in_w + h * in_w + w, index)
    offset = nc * out_d * out_h * out_w + linear
    tl.store(out_ptr + offset, best, mask)
    if WRITE_IDX:
        tl.store(idx_ptr + offset, index.to(tl.int64), mask)
_fractional_max_pool3d_general_kernel = _pool_flat_kernel

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
