REFERENCE_DEVICE = 'target'
import math
import torch.nn.functional as F

import torch
def run(B, x, dt, dA_cumsum):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups

    x_c = x.reshape(batch, nchunks, chunk_size, nheads, headdim).float()
    B_c = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).float()
    B_c = B_c.repeat_interleave(ratio, dim=3)

    dA_last = dA_cumsum[..., -1:].float()
    decay = torch.exp(dA_last - dA_cumsum.float())
    scale = (decay * dt.float()).permute(0, 2, 3, 1)

    Bs = B_c * scale.unsqueeze(-1)
    states = torch.einsum("bcthp,bcthn->bchpn", x_c, Bs)
    return states
def _build_case(batch, nchunks, chunk_size, nheads, ngroups, headdim, dstate, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    seqlen = nchunks * chunk_size

    x = torch.randn(
        batch, seqlen, nheads, headdim, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    b = torch.randn(
        batch, seqlen, ngroups, dstate, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    raw_dt = torch.rand(batch, seqlen, nheads, generator=g, device=_DEVICE, dtype=torch.float32)
    a = -torch.rand(nheads, generator=g, device=_DEVICE, dtype=torch.float32) - 0.1
    dt_out, dA_cumsum = chunk_cumsum_reference(raw_dt, a, chunk_size)

    return dict(B=b, x=x, dt=dt_out, dA_cumsum=dA_cumsum, check=_check)

def _check(actual, expected):
    # The baseline casts the decay-scaled B tile back to bf16 before the
    # tl.dot accumulation (matching x's dtype), so this is bf16-precision
    # matmul, not a full fp32 computation -- use bf16 tolerance even though
    # the output tensor itself is float32.
    assert_close(actual, expected, atol=3e-2, rtol=3e-2)

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
    parameters = {'B', 'x', 'dt', 'dA_cumsum'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

