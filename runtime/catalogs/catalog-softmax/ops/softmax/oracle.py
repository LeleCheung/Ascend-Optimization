import torch


REFERENCE_DEVICE = 'target'


def gen_inputs(ctx, device):
    inputs = ctx['inputs']
    if 'case' not in inputs:
        return None
    case = inputs['case']
    shape = tuple(case['shape'])
    dtype = getattr(torch, case['dtype'])
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError, ValueError):
        generator = torch.Generator(device='cpu')
    generator.manual_seed(ctx['seed'])
    factory_device = generator.device
    inp = torch.randn(shape, dtype=dtype, device=factory_device, generator=generator)
    if generator.device.type != torch.device(device).type:
        inp = inp.to(device)
    if case['neg_inf']:
        inp = torch.where(inp < 0.0, float('-inf'), inp)
    return {'self': inp, 'dim': case['dim']}


def correctness_run(self, dim, half_to_float=False):
    ref = torch.nn.functional.softmax(self.to(torch.float64), dim=dim)
    return ref.to(torch.float32 if half_to_float else self.dtype)


def timing_run(self, dim, half_to_float=False):
    out = torch.nn.functional.softmax(self, dim=dim)
    if half_to_float:
        out = out.to(torch.float32)
    return out
