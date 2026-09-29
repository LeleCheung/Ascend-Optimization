REFERENCE_DEVICE = 'target'
import math
import torch.nn.functional as F

import torch
def run(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape

    if initial_states is None:
        cur = states.new_zeros(batch, nheads, dim, dtype=torch.float32)
    else:
        cur = initial_states.float().clone()

    out = torch.empty(batch, nchunks, nheads, dim, device=states.device, dtype=states.dtype)
    states_f = states.float()
    dA_last = dA_cumsum[..., -1].float().permute(0, 2, 1)

    for c in range(nchunks):
        out[:, c] = cur.to(states.dtype)
        decay = torch.exp(dA_last[:, c]).unsqueeze(-1)
        cur = cur * decay + states_f[:, c]

    final_states = cur
    return out, final_states
def _build_case(batch, nchunks, chunk_size, nheads, dim, has_init=False, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    states = torch.randn(
        batch, nchunks, nheads, dim, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    raw_dt = torch.rand(
        batch, nchunks * chunk_size, nheads, generator=g, device=_DEVICE, dtype=torch.float32
    )
    a = -torch.rand(nheads, generator=g, device=_DEVICE, dtype=torch.float32) - 0.1
    _, dA_cumsum = chunk_cumsum_reference(raw_dt, a, chunk_size)
    initial_states = None
    if has_init:
        initial_states = torch.randn(
            batch, nheads, dim, generator=g, device=_DEVICE, dtype=torch.float32
        ).to(dtype)
    return dict(states=states, dA_cumsum=dA_cumsum, initial_states=initial_states, check=_check)

def _check(actual, expected):
    a_out, a_final = actual
    e_out, e_final = expected
    assert_close(a_out, e_out)
    assert_close(a_final, e_final, dtype=torch.float32)

def chunk_cumsum_reference(dt, A, chunk_size, dt_bias=None, dt_softplus=False):
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
    parameters = {'states', 'dA_cumsum', 'initial_states'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

