import logging
import math
import struct
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
logger = logging.getLogger(__name__)

@libentry()
@triton.jit(do_not_specialize=['scale', 'inv_scale', 'zero_point', 'quant_min', 'quant_max'])
def fake_quantize_per_tensor_affine_cachemask_kernel(input_ptr, output_ptr, mask_ptr, n_elements, scale, inv_scale, zero_point, quant_min, quant_max, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=active).to(tl.float32)
    scale = scale.to(tl.float32)
    inv_scale = inv_scale.to(tl.float32)
    quantized = tl_extra_shim.nearbyint(value * inv_scale) + zero_point
    cachemask = (quantized >= quant_min) & (quantized <= quant_max)
    quantized = tl.minimum(tl.maximum(quantized, quant_min), quant_max)
    output = (quantized - zero_point) * scale
    tl.store(output_ptr + offsets, output, mask=active)
    tl.store(mask_ptr + offsets, cachemask, mask=active)

def _validate_and_convert_qparams(scale, zero_point, quant_min, quant_max):
    if quant_min > quant_max:
        raise RuntimeError('quant_min must be less than or equal to quant_max')
    if zero_point < quant_min or zero_point > quant_max:
        raise RuntimeError('zero_point must be between quant_min and quant_max')
    scale = struct.unpack('f', struct.pack('f', float(scale)))[0]
    if scale == 0.0:
        inv_scale = math.copysign(math.inf, scale)
    else:
        inv_scale = struct.unpack('f', struct.pack('f', 1.0 / scale))[0]
    return (scale, inv_scale)

def _launch(input, output, cachemask, scale, zero_point, quant_min, quant_max):
    scale, inv_scale = _validate_and_convert_qparams(scale, zero_point, quant_min, quant_max)
    contiguous_input = _strict_contiguous(input)
    n_elements = contiguous_input.numel()
    if n_elements == 0:
        return
    contiguous_output = output
    if not output.is_contiguous():
        contiguous_output = _strict_empty_like(contiguous_input)
    contiguous_mask = cachemask
    if not cachemask.is_contiguous():
        contiguous_mask = _strict_empty_like(contiguous_input, dtype=torch.bool)
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(input.device):
        fake_quantize_per_tensor_affine_cachemask_kernel[grid](contiguous_input, contiguous_output, contiguous_mask, n_elements, scale, inv_scale, int(zero_point), int(quant_min), int(quant_max), BLOCK_SIZE=block_size)
    if contiguous_output is not output:
        _strict_copy_into(output, contiguous_output)
    if contiguous_mask is not cachemask:
        _strict_copy_into(cachemask, contiguous_mask)

def fake_quantize_per_tensor_affine_cachemask(input, scale, zero_point, quant_min, quant_max):
    logger.debug('GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK')
    contiguous_input = _strict_contiguous(input)
    output = _strict_empty_like(contiguous_input)
    cachemask = _strict_empty_like(contiguous_input, dtype=torch.bool)
    _launch(contiguous_input, output, cachemask, scale, zero_point, quant_min, quant_max)
    return (output, cachemask)

def fake_quantize_per_tensor_affine_cachemask_out(input, scale, zero_point, quant_min, quant_max, *, out0, out1):
    logger.debug('GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_OUT')
    if out0.dtype != input.dtype:
        raise RuntimeError(f'Expected out tensor to have dtype {input.dtype}, but got {out0.dtype} instead')
    if out1.dtype != torch.bool:
        raise RuntimeError(f'Expected out tensor to have dtype bool, but got {out1.dtype} instead')
    if out0.device != input.device or out1.device != input.device:
        raise RuntimeError('input and out tensors must be on the same device')
    out0.resize_(input.shape)
    out1.resize_(input.shape)
    _launch(input, out0, out1, scale, zero_point, quant_min, quant_max)
    return (out0, out1)
run = fake_quantize_per_tensor_affine_cachemask_out

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
