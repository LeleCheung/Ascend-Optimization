"""夜间体检的独立 PyTorch 参考与输入构建；不进入候选源码。"""
import torch
import torch.nn.functional as F

REFERENCE_DEVICE = 'target'
OPERATOR = 'selu_'


def gen_inputs(ctx, device):
    s = ctx['inputs']['spec']
    dtype = getattr(torch, s.get('dtype', 'float32'))
    g = torch.Generator().manual_seed(ctx.get('seed', 0))
    def random(shape):
        return torch.randn(shape, generator=g).to(dtype=dtype, device=device)
    op = OPERATOR
    if op == 'value_selecting_reduction_backward':
        sizes, dim, keep = s['sizes'], s['dim'], s['keepdim']
        axis = dim % len(sizes)
        shape = list(sizes)
        if keep:
            shape[axis] = 1
        else:
            shape.pop(axis)
        grad = random(shape)
        indices = torch.randint(sizes[axis], shape, generator=g).to(device)
        if s.get('strided') and shape:
            # 相同逻辑shape，但两倍物理跨度；grad与indices均测试stride。
            grad_base = torch.empty((*shape, 2), dtype=dtype, device=device)
            index_base = torch.empty((*shape, 2), dtype=torch.int64, device=device)
            grad_base[..., 0] = grad
            index_base[..., 0] = indices
            grad, indices = grad_base[..., 0], index_base[..., 0]
        return dict(grad=grad, dim=dim, indices=indices, sizes=sizes, keepdim=keep)
    if op == 'scatter_add':
        shape, dim = s['shape'], s['dim']
        x, src = random(shape), random(shape)
        indices = torch.randint(shape[dim], shape, generator=g).to(device)
        return dict(inp=x, dim=dim, index=indices, src=src)
    if op == 'fractional_max_pool3d':
        x = random(s['shape'])
        samples = torch.rand((s['shape'][0], s['shape'][1], 3), generator=g).to(dtype=dtype, device=device)
        return dict(input=x, kernel_size=s['kernel_size'], output_size=s['output_size'],
                    return_indices=True, _random_samples=samples)
    if op == 'upsample_bilinear2d':
        return dict(input=random(s['shape']), output_size=s['output_size'], align_corners=s['align_corners'])
    if op == 'cumulative_trapezoid':
        y = random(s['shape'])
        if s.get('transpose'):
            y = y.transpose(-1, -2)
        return dict(y=y, dx=s['dx'], dim=s['dim'])
    x = random(s['shape'])
    if op == 'conj_physical':
        if dtype.is_complex:
            x = torch.complex(torch.randn(s['shape'], generator=g), torch.randn(s['shape'], generator=g)).to(dtype=dtype, device=device)
        if s.get('transpose'):
            x = x.transpose(-1, -2)
        if s.get('conj'):
            x = x.conj()
        return dict(input=x)
    if op == 'uniform':
        return dict(self=x, from_=s['from_'], to=s['to'])
    if op == 'log10_':
        x = x.abs() + 0.05
    if s.get('strided'):
        base = torch.empty((*x.shape, 2), dtype=dtype, device=device)
        base[..., 0] = x
        x = base[..., 0]
    return dict(A=x)


def run(A):
    kwargs = locals().copy()
    op = OPERATOR
    if op in {'selu_', 'rad2deg_', 'log10_'}:
        a = kwargs['A']
        return {'selu_': torch.selu_, 'rad2deg_': torch.rad2deg_, 'log10_': torch.log10_}[op](a)
    if op == 'value_selecting_reduction_backward':
        return torch.ops.aten.value_selecting_reduction_backward(**kwargs)
    if op == 'cumulative_trapezoid':
        return torch.cumulative_trapezoid(**kwargs)
    if op == 'scatter_add':
        return torch.scatter_add(kwargs['inp'], kwargs['dim'], kwargs['index'], kwargs['src'])
    if op == 'upsample_bilinear2d':
        return torch.ops.aten.upsample_bilinear2d(**kwargs)
    if op == 'fractional_max_pool3d':
        return F.fractional_max_pool3d(**kwargs)
    if op == 'conj_physical':
        return torch.conj_physical(kwargs['input'])
    if op == 'uniform':
        return torch.ops.aten.uniform(kwargs['self'], kwargs['from_'], kwargs['to'])
    raise ValueError(op)



import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/cann-priority-20261102/references/artifacts/_hgkda_cann_priority_v1.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='50c322b97e70748ff21501eedc4bbf730c6df2c9ea03871f5c26a91ac4aaed12'
s=importlib.util.spec_from_file_location('_hgkda_cann_priority_v1',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
assert torch.__version__=='2.9.0+cpu' and torch._C._GLIBCXX_USE_CXX11_ABI==True
def run(A):
    return _cpp.inplace(A,0)
