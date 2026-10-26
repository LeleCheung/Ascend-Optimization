#!/usr/bin/env python3
"""补充公平对照的纯NPU分解；非torch_npu官方单算子，不覆盖原参考。"""
import torch


def conjugate(input):
    """复制实数；复制复数存储并按逻辑共轭位翻转虚部。"""
    if not input.is_complex():
        return input.contiguous().clone()
    raw = input.conj() if input.is_conj() else input
    out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    src, dst = torch.view_as_real(raw), torch.view_as_real(out)
    dst.copy_(src)
    if not input.is_conj():
        dst[...,1].neg_()
    return out


def triple(value):
    if isinstance(value, int):
        return (value,)*3
    if len(value)==1:
        return tuple(value)*3
    return tuple(value)


def fractional_pool(input, kernel_size, output_size=None, output_ratio=None,
                    return_indices=True, _random_samples=None):
    """按固定PyTorch CPU算法展开采样、gather与严格有序最大值选择。"""
    x = input
    squeeze = x.ndim==4
    if squeeze:
        x=x.unsqueeze(0)
    assert x.ndim==5
    n,c,d,h,w=x.shape
    kd,kh,kw=triple(kernel_size)
    if output_size is None:
        ratios=(output_ratio,)*3 if isinstance(output_ratio,float) else tuple(output_ratio)
        output_size=[int(size*ratio) for size,ratio in zip((d,h,w),ratios)]
    od,oh,ow=triple(output_size)
    samples=_random_samples
    if samples is None:
        samples=torch.rand((n,c,3),dtype=x.dtype,device=x.device)
    if samples.ndim==2:
        samples=samples.unsqueeze(0)
    starts=[]
    for axis,(size,outsize,pool) in enumerate(zip((d,h,w),(od,oh,ow),(kd,kh,kw))):
        sample=samples[:n,:,axis].unsqueeze(-1)
        if outsize==1:
            start=torch.full((n,c,1),size-pool,dtype=torch.int64,device=x.device)
        else:
            # Half/BFloat16在CPU原算法每次scalar_t运算后舍入；NPU显式保持该dtype。
            a=torch.full((n,c,1),size-pool,dtype=x.dtype,device=x.device)
            divisor=torch.full((n,c,1),outsize-1,dtype=x.dtype,device=x.device)
            alpha=a/divisor
            i=torch.arange(outsize,dtype=torch.int64,device=x.device).to(x.dtype)
            shifted=i+sample
            scaled=shifted*alpha
            base=sample*alpha
            start=scaled.to(torch.int64)-base.to(torch.int64)
            tail=torch.arange(outsize,device=x.device).reshape(1,1,outsize)==outsize-1
            start=torch.where(tail,torch.full_like(start,size-pool),start)
        starts.append(start)
    ds=starts[0][..., :,None,None]
    hs=starts[1][..., None,:,None]
    ws=starts[2][..., None,None,:]
    first=ds*h*w+hs*w+ws
    best=torch.full((n,c,od,oh,ow),-float('inf'),dtype=x.dtype,device=x.device)
    best_index=first.clone()
    flat=x.contiguous().reshape(n,c,d*h*w)
    for di in range(kd):
        for hi in range(kh):
            for wi in range(kw):
                index=(ds+di)*h*w+(hs+hi)*w+ws+wi
                value=torch.gather(flat,2,index.reshape(n,c,-1)).reshape(n,c,od,oh,ow)
                take=(value>best)|torch.isnan(value)
                best=torch.where(take,value,best)
                best_index=torch.where(take,index,best_index)
    if squeeze:
        best,best_index=best.squeeze(0),best_index.squeeze(0)
    return (best,best_index) if return_indices else best
