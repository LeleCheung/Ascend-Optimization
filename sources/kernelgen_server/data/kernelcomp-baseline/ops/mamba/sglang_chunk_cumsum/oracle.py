REFERENCE_DEVICE = 'target'

import math
import torch
import torch.nn.functional as F
def run(dt, A, chunk_size, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = torch.where(dt_f <= 20.0, F.softplus(dt_f), dt_f)
    dt_f = dt_f.clamp(min=0.0)

    dt_out = dt_f.reshape(batch, nchunks, chunk_size, nheads).permute(0, 3, 1, 2).contiguous()
    dA = dt_out * A.float().view(1, nheads, 1, 1)
    dA_cumsum = dA.cumsum(dim=-1)
    return dt_out, dA_cumsum
def _build_case(
    batch, nchunks, chunk_size, nheads, dt_bias=False, dt_softplus=False, dtype=torch.bfloat16, seed=0
):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    seqlen = nchunks * chunk_size
    dt = torch.randn(batch, seqlen, nheads, generator=g, device=_DEVICE, dtype=torch.float32).to(
        dtype
    )
    a = -torch.rand(nheads, generator=g, device=_DEVICE, dtype=torch.float32) - 0.1
    bias = None
    if dt_bias:
        bias = torch.randn(nheads, generator=g, device=_DEVICE, dtype=torch.float32)
    return dict(
        dt=dt, A=a, chunk_size=chunk_size, dt_bias=bias, dt_softplus=dt_softplus, check=_check
    )

def _check(actual, expected):
    a_dt, a_dA = actual
    e_dt, e_dA = expected
    assert_close(a_dt, e_dt, dtype=torch.float32)
    assert_close(a_dA, e_dA, dtype=torch.float32)




_DEVICE = None


def _materialize_arg(value, device):
    if isinstance(value, dict) and "__tensor__" in value:
        dtype = value.get("dtype", "float32")
        return torch.tensor(
            value["__tensor__"],
            dtype=getattr(torch, dtype),
            device=device,
        )
    return value


def gen_inputs(ctx, device):
    global _DEVICE
    _DEVICE = device
    case_args = ctx["inputs"].get("_case_args", {})
    args = [_materialize_arg(value, device) for value in case_args.get("args", [])]
    kwargs = {
        key: _materialize_arg(value, device)
        for key, value in case_args.get("kwargs", {}).items()
    }
    dtype = kwargs.get("dtype")
    if isinstance(dtype, str) and dtype:
        kwargs["dtype"] = getattr(torch, dtype)
    constructor = case_args.get("constructor", "_case")
    if constructor == '_case':
        built = _build_case(*args, **kwargs)
    parameters = {'dt', 'A', 'chunk_size', 'dt_bias', 'dt_softplus'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

