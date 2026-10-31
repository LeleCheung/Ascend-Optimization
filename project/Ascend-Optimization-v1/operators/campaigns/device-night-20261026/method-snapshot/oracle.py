"""设备机制实验的独立输入与PyTorch参考；不进入候选源码。"""
import torch

REFERENCE_DEVICE='target'
OPERATOR='__OPERATOR__'


def gen_inputs(ctx,device):
    s=ctx['inputs']['spec'];dtype=getattr(torch,s['dtype'])
    generator=torch.Generator().manual_seed(ctx.get('seed',71329))
    def random(shape):
        x=torch.randn(shape,generator=generator).to(dtype=dtype,device=device)
        if s.get('strided'):
            storage=torch.empty((*shape,2),dtype=dtype,device=device)
            storage[...,0]=x;x=storage[...,0]
        return x
    if OPERATOR=='matmul_bias_activation':
        m,n,k=s['shape']
        return dict(input=random([m,k]),weight=random([k,n]),bias=random([1,n] if s.get('bias_row') else [n]))
    return dict(inp=random(s['shape']),dim=s.get('dim',1),keepdim=s.get('keepdim',False))


def run(**inputs):
    if OPERATOR=='matmul_bias_activation':
        return torch.relu(inputs['input']@inputs['weight']+inputs['bias'])
    return torch.amin(inputs['inp'],dim=inputs['dim'],keepdim=inputs['keepdim'])
