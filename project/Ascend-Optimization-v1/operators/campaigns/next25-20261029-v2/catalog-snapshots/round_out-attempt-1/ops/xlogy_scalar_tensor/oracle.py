"""新目标的独立PyTorch输入和参考；run签名由冻结入口生成。"""
import math
import torch
import torch.nn.functional as F

REFERENCE_DEVICE = 'target'
OPERATOR = 'xlogy_scalar_tensor'


def gen_inputs(ctx, device):
    s = ctx['inputs']['spec']
    dtype = getattr(torch, s['dtype'])
    gen = torch.Generator().manual_seed(ctx.get('seed', 0))
    def random(shape, positive=False):
        x = torch.randn(shape, generator=gen)
        if positive:
            x = x.abs()+0.05
        return x.to(device=device, dtype=dtype)
    x = random(s['shape'], OPERATOR in {'log2_', 'log10_', 'xlogy_scalar_tensor', 'special_xlogy_'})
    if s.get('special') and x.numel() >= 7:
        edge = [0., -0., float('nan'), float('inf'), -float('inf'), -1., 1.]
        if OPERATOR not in {'cross_out', 'fmod_tensor_', 'round_out', 'special_round_out', 'fake_quantize_per_tensor_affine_cachemask_out'}:
            x.flatten()[:7] = torch.tensor(edge, dtype=dtype, device=device)
    if s.get('strided'):
        base = torch.empty((*x.shape, 2), dtype=dtype, device=device)
        base[..., 0] = x
        x = base[..., 0]
    op = OPERATOR
    if op == 'native_group_norm':
        n, c = x.shape[:2]
        return dict(input=x, weight=random([c]) if s['affine'] else None, bias=random([c]) if s['affine'] else None,
                    N=n, C=c, HxW=math.prod(x.shape[2:]), group=s['groups'], eps=1.e-5)
    if op == 'linalg_inv_ex':
        x = random(s['shape'])
        x = x + torch.eye(x.shape[-1], device=device, dtype=dtype)*(x.shape[-1]+1)
        if s.get('singular'):
            x[..., -1, :] = x[..., 0, :]
        return dict(A=x, check_errors=False)
    if op == 'cross_out':
        return dict(input=x, other=random(s['shape']), dim=s['dim'], out=torch.empty_like(x))
    if op == 'xlogy_scalar_tensor':
        return dict(self=0. if s.get('strided') else 2., other=x)
    if op == 'nan_to_num_':
        return dict(A=x, nan=0., posinf=None, neginf=None)
    if op == 'hardshrink_backward':
        return dict(grad_out=random(s['shape']), self_t=x, lambd=0.5)
    if op == 'subtract_':
        return dict(A=x, B=random(s['shape']), alpha=0.5)
    if op == 'fix':
        return dict(self=x)
    if op == 'round_out':
        return dict(input=x, decimals=1 if s.get('strided') else 0, out=torch.empty_like(x))
    if op == 'special_round_out':
        return dict(self=x, out=torch.empty_like(x), decimals=1 if s.get('strided') else 0)
    if op in {'hardswish', 'hardswish_out'}:
        return dict(x=x, **(dict(out=torch.empty_like(x)) if op.endswith('_out') else {}))
    if op == 'fake_quantize_per_tensor_affine_cachemask_out':
        return dict(input=x, scale=0.1, zero_point=3, quant_min=0, quant_max=255,
                    out0=torch.empty_like(x), out1=torch.empty_like(x, dtype=torch.bool))
    if op == 'special_xlogy_':
        return dict(A=x, B=random(s['shape'], True))
    if op == 'fmod_tensor_':
        return dict(A=x, B=random(s['shape'], True))
    return dict(A=x)


def run(self, other):
    kwargs = locals().copy()
    op = OPERATOR
    if op in {'negative_', 'expm1_', 'log2_', 'log10_', 'rad2deg_', 'selu_'}:
        a = kwargs['A']
        return getattr(a, 'neg_' if op=='negative_' else op)()
    if op == 'xlogy_scalar_tensor':
        return torch.xlogy(kwargs['self'], kwargs['other'])
    if op == 'nan_to_num_':
        return kwargs['A'].nan_to_num_(kwargs['nan'], kwargs['posinf'], kwargs['neginf'])
    if op == 'hardshrink_backward':
        return torch.ops.aten.hardshrink_backward(kwargs['grad_out'], kwargs['self_t'], kwargs['lambd'])
    if op == 'subtract_':
        return kwargs['A'].sub_(kwargs['B'], alpha=kwargs['alpha'])
    if op == 'native_group_norm':
        return torch.native_group_norm(kwargs['input'], kwargs['weight'], kwargs['bias'], kwargs['N'], kwargs['C'], kwargs['HxW'], kwargs['group'], kwargs['eps'])
    if op == 'fix':
        return torch.fix(kwargs['self'])
    if op == 'cross_out':
        return torch.cross(kwargs['input'], kwargs['other'], dim=kwargs['dim'], out=kwargs['out'])
    if op in {'round_out', 'special_round_out'}:
        return torch.round(kwargs.get('input', kwargs.get('self')), decimals=kwargs['decimals'], out=kwargs['out'])
    if op == 'hardswish':
        return F.hardswish(kwargs['x'])
    if op == 'hardswish_out':
        return torch.ops.aten.hardswish.out(kwargs['x'], out=kwargs['out'])
    if op == 'fake_quantize_per_tensor_affine_cachemask_out':
        return torch.ops.aten.fake_quantize_per_tensor_affine_cachemask.out(kwargs['input'], kwargs['scale'], kwargs['zero_point'], kwargs['quant_min'], kwargs['quant_max'], out0=kwargs['out0'], out1=kwargs['out1'])
    if op == 'special_xlogy_':
        return kwargs['A'].xlogy_(kwargs['B'])
    if op == 'fmod_tensor_':
        return kwargs['A'].fmod_(kwargs['B'])
    if op == 'linalg_inv_ex':
        return torch.linalg.inv_ex(kwargs['A'], check_errors=kwargs['check_errors'])
    raise ValueError(op)
