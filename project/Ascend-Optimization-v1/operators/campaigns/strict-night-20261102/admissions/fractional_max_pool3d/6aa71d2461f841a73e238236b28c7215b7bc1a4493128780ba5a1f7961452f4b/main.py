import torch
import triton
import triton.language as tl

_PACK_MASK = 16383


@triton.jit
def _round_coordinate(value, DT: tl.constexpr):
    """Round a coordinate expression to the scalar dtype used by the reference."""
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
def _alpha_of(in_size, out_size, K: tl.constexpr, DT: tl.constexpr):
    return _round_coordinate(
        _round_coordinate(tl.cast(in_size - K, tl.float32), DT)
        / _round_coordinate(tl.cast(tl.maximum(out_size - 1, 1), tl.float32), DT),
        DT)


@triton.jit
def _start_of(index, sample, alpha, K: tl.constexpr, DT: tl.constexpr):
    base = _round_coordinate(sample * alpha, DT).to(tl.int32)
    shifted = _round_coordinate(tl.cast(index, tl.float32) + sample, DT)
    return _round_coordinate(shifted * alpha, DT).to(tl.int32) - base


@triton.jit
def _pool_tile_kernel(in_ptr, out_ptr, idx_ptr, samp_ptr, pack_a, pack_b,
                      OH: tl.constexpr, OW: tl.constexpr,
                      KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                      KH2: tl.constexpr, KW2: tl.constexpr,
                      WRITE_IDX: tl.constexpr):
    """Windows that tile the input exactly (in_x == K_x * out_x).

    Every pooling tap is an affine 2D block load over the output grid, so loads
    stay contiguous/strided DMA instead of per-lane gathers.  The reference
    window start is reproduced with the same rounding emulation; its deviation
    from the nominal K*i grid is at most one element (host gate), which the K+2
    candidate taps with per-lane masks cover exactly.
    """
    in_w = pack_a & 16383
    in_c = (pack_a >> 14) & 16383
    in_d = (pack_a >> 28) & 16383
    in_h = (pack_a >> 42) & 16383
    out_d = pack_b & 16383
    out_h = (pack_b >> 14) & 16383
    out_w = (pack_b >> 28) & 16383
    pid = tl.program_id(0)
    nc = pid // out_d
    do = pid % out_d
    ni = nc // in_c
    ci = nc % in_c
    samp = samp_ptr + ni * (3 * in_c) + ci * 3
    sample_d = tl.load(samp)
    sample_h = tl.load(samp + 1)
    sample_w = tl.load(samp + 2)
    dt: tl.constexpr = samp_ptr.dtype.element_ty
    alpha_d = _alpha_of(in_d, out_d, KD, dt)
    alpha_h = _alpha_of(in_h, out_h, KH, dt)
    alpha_w = _alpha_of(in_w, out_w, KW, dt)
    d_start = _start_of(do, sample_d, alpha_d, KD, dt)
    d_start = tl.where(do == out_d - 1, in_d - KD, d_start)
    ho = tl.arange(0, OH)
    wo = tl.arange(0, OW)
    h_start = _start_of(ho, sample_h, alpha_h, KH, dt)
    w_start = _start_of(wo, sample_w, alpha_w, KW, dt)
    h_start = tl.where(ho == out_h - 1, in_h - KH, h_start)
    w_start = tl.where(wo == out_w - 1, in_w - KW, w_start)
    e_h = h_start - ho * KH
    e_w = w_start - wo * KW
    oh_ok = ho < out_h
    ow_ok = wo < out_w
    row0 = ho * KH
    col0 = wo * KW
    in_hw = in_h * in_w
    best = tl.full((OH, OW), float('-inf'), in_ptr.dtype.element_ty)
    bidx = tl.zeros((OH, OW), tl.int32)
    nc_base = (ni * in_c + ci) * in_d * in_hw
    for kd in tl.static_range(KD):
        doff = (d_start + kd) * in_hw
        for jh in tl.static_range(KH2):
            sh = jh - 1
            mh = (sh >= e_h) & (sh < e_h + KH) & oh_ok
            hrow = row0 + sh
            for jw in tl.static_range(KW2):
                sw = jw - 1
                m = mh[:, None] & ((sw >= e_w) & (sw < e_w + KW) & ow_ok)[None, :]
                off = doff + hrow[:, None] * in_w + (col0[None, :] + sw)
                v = tl.load(in_ptr + nc_base + off, mask=m, other=float('-inf'))
                take = (v > best) | (v != v)
                best = tl.where(take, v, best)
                if WRITE_IDX:
                    bidx = tl.where(take, off, bidx)
    off_out = ((nc * out_d + do) * out_h + ho[:, None]) * out_w + wo[None, :]
    mstore = oh_ok[:, None] & ow_ok[None, :]
    tl.store(out_ptr + off_out, best, mask=mstore)
    if WRITE_IDX:
        tl.store(idx_ptr + off_out, bidx.to(tl.int64), mask=mstore)


@triton.jit
def _pool_flat_kernel(in_ptr, out_ptr, idx_ptr, samp_ptr, s_n, s_c, s_d, s_h, s_w, ss_n, ss_c, in_c, in_d, in_h, in_w, out_d, out_h, out_w, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, BLOCK_W: tl.constexpr, WRITE_IDX: tl.constexpr):
    """Fallback gather kernel: any strides, any window geometry."""
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
    ad = _alpha_of(in_d, out_d, KD, dt)
    ah = _alpha_of(in_h, out_h, KH, dt)
    aw = _alpha_of(in_w, out_w, KW, dt)
    ds = _start_of(do, sd, ad, KD, dt)
    hs = _start_of(ho, sh, ah, KH, dt)
    ws = _start_of(wo, sw, aw, KW, dt)
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


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 1:
        return (value[0], value[0], value[0])
    return (value[0], value[1], value[2])


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length() if n > 0 else 1


def _relerr(dtype):
    if dtype == torch.float16:
        return 2.0 ** -10
    if dtype == torch.bfloat16:
        return 2.0 ** -7
    if dtype == torch.float32:
        return 2.0 ** -23
    if dtype == torch.float64:
        return 2.0 ** -52
    return 1.0


def _pack(a, b):
    return a | (b << 14)


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
    k_max = max(kernel_d, kernel_h, kernel_w)
    in_max = max(in_d, in_h, in_w)
    tile_ok = (
        in_d == kernel_d * out_d and in_h == kernel_h * out_h and in_w == kernel_w * out_w
        and s_w == 1 and s_h == in_w and s_d == in_h * in_w
        and _random_samples.stride(0) == 3 * in_c and _random_samples.stride(1) == 3
        and 1 <= kernel_d <= 8 and 1 <= kernel_h <= 8 and 1 <= kernel_w <= 8
        and in_max <= _PACK_MASK and in_c <= _PACK_MASK and out_d <= _PACK_MASK
        and out_h <= _PACK_MASK and out_w <= _PACK_MASK
        and _next_pow2(out_h) * _next_pow2(out_w) <= 16384
        and input.numel() <= 2 ** 30
    )
    if tile_ok:
        # The reference start index is floor(round((i + sample) * alpha)) minus
        # floor(round(sample * alpha)) in the scalar dtype.  Against the nominal
        # K*i grid each floor can move by at most one and their difference by at
        # most one, provided the total rounding error stays below 1:
        # in_max * rel * (K + 2) / 2 < 1, rel being the dtype ulp step.
        tile_ok = in_max * _relerr(input.dtype) * (k_max + 2) < 2.0
    if tile_ok:
        pack_a = _pack(in_w, _pack(in_c, _pack(in_d, in_h)))
        pack_b = _pack(out_d, _pack(out_h, out_w))
        _pool_tile_kernel[(in_n * in_c * out_d,)](
            input, output, indices, _random_samples, pack_a, pack_b,
            _next_pow2(out_h), _next_pow2(out_w),
            kernel_d, kernel_h, kernel_w, kernel_h + 2, kernel_w + 2, return_indices)
    else:
        block_w = 128
        grid = (in_n * in_c, triton.cdiv(out_d * out_h * out_w, block_w))
        _pool_flat_kernel[grid](
            input, output, indices, _random_samples,
            s_n, s_c, s_d, s_h, s_w, _random_samples.stride(0), _random_samples.stride(1),
            in_c, in_d, in_h, in_w, out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w, block_w, return_indices)
    if squeeze:
        output = output.squeeze(0)
        if return_indices:
            indices = indices.squeeze(0)
    return (output, indices) if return_indices else output


run = fractional_max_pool3d


"""候选内嵌的Triton存储操作：复制、连续化、填充、转换均执行Triton核。"""


@triton.jit
def _strict_copy_kernel(X, Y, N: tl.constexpr, SHAPE: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remain = i
    xp = tl.full((BLOCK,), 0, tl.int64)
    yp = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remain % SHAPE[d]
        remain = remain // SHAPE[d]
        xp += coord * XS[d]
        yp += coord * YS[d]
    value = tl.load(X + xp, i < N, other=0)
    tl.store(Y + yp, value, i < N)


@triton.jit
def _strict_fill_kernel(Y, N: tl.constexpr, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, VALUE, i < N)


@triton.jit
def _strict_random_kernel(Y, N: tl.constexpr, SEED, OFFSET, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # 备用随机采样路径；全部随机数计算在Triton内，生成器仅在宿主管理状态。
    value = tl.rand(SEED, (i.to(tl.uint64) + OFFSET).to(tl.uint32))
    tl.store(Y + i, value, i < N)


def _strict_empty_like(x, *, dtype=None, **kwargs):
    return torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype if dtype is None else dtype)


def _strict_copy_into(dst, src):
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError('strict copy requires equal shapes')
    if dst.numel():
        _strict_copy_kernel[(triton.cdiv(dst.numel(), 1024),)](src, dst, dst.numel(), tuple(dst.shape), tuple(src.stride()), tuple(dst.stride()), 1024)
    return dst


def _strict_clone(x, **kwargs):
    return _strict_copy_into(_strict_empty_like(x), x)


def _strict_contiguous(x, *args, **kwargs):
    return x if x.is_contiguous() else _strict_clone(x)


def _strict_cast(x, dtype):
    if dtype == x.dtype:
        return x
    return _strict_copy_into(_strict_empty_like(x, dtype=dtype), x)


def _strict_reshape(x, *shape):
    # view只操作元数据，输入连续化由上面的Triton复制完成。
    return _strict_contiguous(x).view(*shape)


def _strict_flatten(x, start_dim=0, end_dim=-1):
    rank = x.ndim
    first = start_dim % rank
    last = end_dim % rank
    size = 1
    for n in x.shape[first:last + 1]:
        size *= n
    shape = tuple(x.shape[:first]) + (size,) + tuple(x.shape[last + 1:])
    return _strict_contiguous(x).view(shape)


def _strict_full(shape, value, **kwargs):
    if isinstance(shape, int):
        shape = (shape,)
    out = torch.empty(shape, **kwargs)
    if out.numel():
        _strict_fill_kernel[(triton.cdiv(out.numel(), 1024),)](out, out.numel(), value, 1024)
    return out


def _strict_zeros(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    return _strict_full(shape, 0, **kwargs)


def _strict_ones(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    return _strict_full(shape, 1, **kwargs)


def _strict_rand(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    generator = kwargs.pop('generator', None)
    out = torch.empty(shape, **kwargs)
    if out.numel():
        if generator is None:
            import torch_npu
            generator = torch.npu.default_generators[out.device.index or 0]
        seed = generator.initial_seed()
        offset = generator.get_offset()
        advance = triton.cdiv(out.numel(), 4) * 4
        if offset + advance >= 2 ** 32:
            raise RuntimeError('备用随机路径超过32位计数范围，明确拒绝而非静默重复')
        generator.set_offset(offset + advance)
        _strict_random_kernel[(triton.cdiv(out.numel(), 256),)](out, out.numel(), seed, offset, 256)
    return out
