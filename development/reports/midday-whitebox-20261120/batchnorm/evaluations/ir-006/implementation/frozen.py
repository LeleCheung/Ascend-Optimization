import torch
import torch_npu
import triton
import triton.language as tl
from torch import Tensor

def _bn_block_m(args):
    return min(64, triton.next_power_of_2(args['batch_dim']))

def _bn_block_n(args):
    block_m = _bn_block_m(args)
    block_n = triton.next_power_of_2(args['spatial_dim'])
    return min(block_n, max(1, 2**10 // block_m))

def _bn_even(args):
    # 当batch与spatial都被tile整除时，逐tile掩码恒为真：用编译期常量分支省掉谓词。
    # 不整除的形状仍走原掩码路径，语义不变。
    block_m = _bn_block_m(args)
    block_n = _bn_block_n(args)
    return args['batch_dim'] % block_m == 0 and args['spatial_dim'] % block_n == 0

_BN_HEURISTICS = {'BLOCK_M': _bn_block_m, 'BLOCK_N': _bn_block_n, 'EVEN': _bn_even}
_BN_CONFIGS = [triton.Config({}, num_warps=2, num_stages=1)]

def _libentry(*args, **kwargs):
    def _decorate(fn):
        return fn
    return _decorate

def make_3d_for_bn(input: Tensor) -> Tensor:
    """
    Converts the input to a 3D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = _strict_flatten(input, 2, -1)
    return input

@_libentry()
@triton.autotune(configs=_BN_CONFIGS, key=['batch_dim', 'spatial_dim'])
@triton.heuristics(_BN_HEURISTICS)
@triton.jit
def batch_norm_with_update_kernel(input_pointer, weight_pointer, bias_pointer, mean_pointer, inv_std_pointer, output_pointer, running_mean_out_pointer, running_var_out_pointer, running_mean_in_pointer, running_var_in_pointer, batch_dim, spatial_dim, input_batch_stride, input_feat_stride, input_spatial_stride, output_batch_stride, output_feat_stride, output_spatial_stride, momentum, eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, EVEN: tl.constexpr):
    feat_pid = tl.program_id(axis=0)
    mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
    n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)
    tile_lane = tl.arange(0, BLOCK_N)
    for m_step in range(0, m_num_steps):
        batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
        input_row_pointer = input_pointer + input_feat_stride * feat_pid + input_batch_stride * batch_offset[:, None]
        # Prime the software pipeline with the first tile of this batch step.
        spatial_offset = tile_lane
        if EVEN:
            # batch_dim % BLOCK_M == 0 and spatial_dim % BLOCK_N == 0, so this
            # tile and every lookahead tile are fully in bounds: no predicate.
            curr_input = tl.load(input_row_pointer + input_spatial_stride * spatial_offset[None, :]).to(tl.float32)
        else:
            batch_mask = batch_offset < batch_dim
            curr_mask = batch_mask[:, None] & (spatial_offset[None, :] < spatial_dim)
            curr_input = tl.load(input_row_pointer + input_spatial_stride * spatial_offset[None, :], mask=curr_mask, other=0.0).to(tl.float32)
        for n_step in range(0, n_num_steps):
            # Issue the next tile's GM load before consuming the current tile so
            # MTE2 can overlap the Welford vector math.
            step = m_step * n_num_steps + n_step + 1
            # Welford recurrence in delta form: one mask select and a scalar
            # reciprocal replace the per-tile vector divide and second select.
            inv_step = 1.0 / step.to(tl.float32)
            spatial_offset = n_step * BLOCK_N + tile_lane
            if EVEN:
                # Clamp the lookahead to the last valid tile so the prefetch
                # never leaves the feature slice; the last value is unused.
                next_spatial_offset = tl.minimum(n_step + 1, n_num_steps - 1) * BLOCK_N + tile_lane
                next_input = tl.load(input_row_pointer + input_spatial_stride * next_spatial_offset[None, :]).to(tl.float32)
                # Mask is constant true: the delta needs no select and the count
                # gains one sample per lane. Numerically identical to the masked
                # form on these lanes.
                delta = curr_input - mean
                cnt += 1
            else:
                # The final lookahead is fully masked (its offsets start at
                # n_num_steps * BLOCK_N >= spatial_dim).
                next_spatial_offset = (n_step + 1) * BLOCK_N + tile_lane
                next_mask = batch_mask[:, None] & (next_spatial_offset[None, :] < spatial_dim)
                next_input = tl.load(input_row_pointer + input_spatial_stride * next_spatial_offset[None, :], mask=next_mask, other=0.0).to(tl.float32)
                # The tile mask is re-derived from the induction variable instead of
                # being carried as a boolean tensor loop argument; the lowering pass
                # rejects carried i1 masks, and this is the same predicate.
                curr_mask = batch_mask[:, None] & (spatial_offset[None, :] < spatial_dim)
                delta = tl.where(curr_mask, curr_input - mean, 0.0)
                cnt += curr_mask.to(tl.int32)
            mean = mean + delta * inv_step
            var = var + delta * delta * (1.0 - inv_step)
            curr_input = next_input
    final_mean = tl.sum(mean * cnt) / (batch_dim * spatial_dim)
    var = tl.sum(var + cnt * (mean - final_mean) * (mean - final_mean)) / (batch_dim * spatial_dim)
    inv_std = tl.math.rsqrt(var + eps)
    mean = final_mean
    tl.store(feat_pid + mean_pointer, mean)
    tl.store(feat_pid + inv_std_pointer, inv_std)
    running_mean_in = tl.load(feat_pid + running_mean_in_pointer)
    running_var_in = tl.load(feat_pid + running_var_in_pointer)
    n = batch_dim * spatial_dim
    new_running_mean = (1 - momentum) * running_mean_in + momentum * mean
    new_running_var = (1 - momentum) * running_var_in + momentum * var * n / (n - 1)
    tl.store(feat_pid + running_mean_out_pointer, new_running_mean)
    tl.store(feat_pid + running_var_out_pointer, new_running_var)
    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0
    for m_step in range(0, m_num_steps):
        batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
        input_row_pointer = input_pointer + input_feat_stride * feat_pid + input_batch_stride * batch_offset[:, None]
        output_row_pointer = output_pointer + output_feat_stride * feat_pid + output_batch_stride * batch_offset[:, None]
        spatial_offset = tile_lane
        if EVEN:
            curr_input = tl.load(input_row_pointer + input_spatial_stride * spatial_offset[None, :]).to(tl.float32)
        else:
            batch_mask = batch_offset < batch_dim
            curr_mask = batch_mask[:, None] & (spatial_offset[None, :] < spatial_dim)
            curr_input = tl.load(input_row_pointer + input_spatial_stride * spatial_offset[None, :], mask=curr_mask, other=0.0).to(tl.float32)
        for n_step in range(0, n_num_steps):
            # Same one-tile lookahead as the statistics pass: the next load is
            # issued before the current tile's scale/shift and store.
            spatial_offset = n_step * BLOCK_N + tile_lane
            if EVEN:
                next_spatial_offset = tl.minimum(n_step + 1, n_num_steps - 1) * BLOCK_N + tile_lane
                next_input = tl.load(input_row_pointer + input_spatial_stride * next_spatial_offset[None, :]).to(tl.float32)
            else:
                next_spatial_offset = (n_step + 1) * BLOCK_N + tile_lane
                next_mask = batch_mask[:, None] & (next_spatial_offset[None, :] < spatial_dim)
                next_input = tl.load(input_row_pointer + input_spatial_stride * next_spatial_offset[None, :], mask=next_mask, other=0.0).to(tl.float32)
                # Re-derive the current tile predicate from n_step rather than
                # carrying an i1 tensor through the loop (unsupported by the
                # Triton-to-Linalg conversion pass on this target).
                curr_mask = batch_mask[:, None] & (spatial_offset[None, :] < spatial_dim)
            output = weight * (curr_input - mean) * inv_std + bias
            if EVEN:
                tl.store(output_row_pointer + output_spatial_stride * spatial_offset[None, :], output)
            else:
                tl.store(output_row_pointer + output_spatial_stride * spatial_offset[None, :], output, mask=curr_mask)
            curr_input = next_input

def _batch_norm_with_update_functional(input: Tensor, weight=None, bias=None, running_mean=None, running_var=None, momentum=0.1, eps=1e-05):
    """Functional batch normalization with running stats update."""
    input_3d = make_3d_for_bn(input)
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = _strict_empty_like(input_3d)
    save_mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    save_invstd = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    running_mean_out = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    running_var_out = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    if running_mean is None:
        running_mean = _strict_zeros(feat_dim, device=input.device, dtype=input.dtype)
    if running_var is None:
        running_var = _strict_ones(feat_dim, device=input.device, dtype=input.dtype)
    batch_norm_with_update_kernel[feat_dim,](input_3d, weight, bias, save_mean, save_invstd, output, running_mean_out, running_var_out, running_mean, running_var, batch_dim, spatial_dim, *input_3d.stride(), *output.stride(), momentum, eps)
    reserve = torch.empty((0,), dtype=torch.uint8, device=input.device)
    return (output.view_as(input), save_mean, save_invstd, reserve, running_mean_out, running_var_out)
run = _batch_norm_with_update_functional

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
