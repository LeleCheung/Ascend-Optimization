"""独立PyTorch参考及CPU随机输入，不被候选导入。"""
import torch
REFERENCE_DEVICE='target'
OPERATOR='asinh_'
def gen_inputs(ctx, device):
    s=ctx['inputs']['spec']
    g=torch.Generator().manual_seed(ctx.get('seed',1729))
    x=torch.randn(s['shape'],generator=g).to(dtype=getattr(torch,s['dtype']),device=device)
    if s.get('strided'):
        base=torch.empty((*x.shape,2),dtype=x.dtype,device=device)
        base[...,0]=x
        x=base[...,0]
    return dict(A=x)
def run(A):
    return getattr(torch,OPERATOR)(A)
