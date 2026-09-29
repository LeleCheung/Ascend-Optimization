import torch

REFERENCE_DEVICE = 'target'


def run(x):
    return x.erfinv_()


def gen_inputs(ctx, device):
    inputs = ctx['inputs']
    case = inputs.get('case')
    if case is None:
        return None
    dtype = torch.__dict__[case['dtype']]
    shape = tuple(case['shape'])
    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator(device='cpu')
        x = torch.empty(shape, dtype=dtype, device='cpu')
        gen.manual_seed(ctx['seed'])
        x.uniform_(-0.99, 0.99, generator=gen)
        return {'x': x.to(device)}
    gen.manual_seed(ctx['seed'])
    x = torch.empty(shape, dtype=dtype, device=device)
    x.uniform_(-0.99, 0.99, generator=gen)
    return {'x': x}
