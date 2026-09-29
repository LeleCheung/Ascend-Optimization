REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F
def run(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    state = state.clone()
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = nheads // ngroups

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = F.softplus(dt_f)

    dA = torch.exp(dt_f.unsqueeze(-1) * A.float().unsqueeze(0))
    B_exp = B.float().repeat_interleave(ratio, dim=1)
    dB = dt_f.unsqueeze(-1) * B_exp.unsqueeze(2)

    new_state = state.float() * dA + dB * x.float().unsqueeze(-1)
    state = new_state.to(state.dtype)

    C_exp = C.float().repeat_interleave(ratio, dim=1)
    y = torch.einsum("bhpn,bhn->bhp", new_state, C_exp)

    if D is not None:
        y = y + D.float() * x.float()
    if z is not None:
        y = y * (z.float() * torch.sigmoid(z.float()))

    return y.to(x.dtype), state
def _build_case(
    batch, nheads, dim, dstate, ngroups=1, has_z=True, dt_softplus=True,
    dtype=torch.bfloat16, seed=0,
):
    # D and dt_bias are always provided (as real tensors, never None): the
    # installed baseline has an argument-packing bug on the `is None` branch
    # for both (`*(...) if x is not None else 0` unpacks a bare int), so
    # this problem's cases avoid that path rather than exercise a bug.
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    state = torch.randn(
        batch, nheads, dim, dstate, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    x = torch.randn(batch, nheads, dim, generator=g, device=_DEVICE, dtype=torch.float32).to(dtype)
    dt = torch.randn(batch, nheads, dim, generator=g, device=_DEVICE, dtype=torch.float32).to(
        dtype
    )
    a = -torch.rand(nheads, dim, dstate, generator=g, device=_DEVICE, dtype=torch.float32) - 0.1
    b = torch.randn(
        batch, ngroups, dstate, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    c = torch.randn(
        batch, ngroups, dstate, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    d = torch.randn(nheads, dim, generator=g, device=_DEVICE, dtype=torch.float32)
    z = None
    if has_z:
        z = torch.randn(batch, nheads, dim, generator=g, device=_DEVICE, dtype=torch.float32).to(
            dtype
        )
    bias = torch.randn(nheads, dim, generator=g, device=_DEVICE, dtype=torch.float32)

    return dict(
        state=state,
        x=x,
        dt=dt,
        A=a,
        B=b,
        C=c,
        D=d,
        z=z,
        dt_bias=bias,
        dt_softplus=dt_softplus,
        check=_check,
    )

def _check(actual, expected):
    a_y, a_state = actual
    e_y, e_state = expected
    assert_close(a_y, e_y)
    assert_close(a_state, e_state)




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
    parameters = {'state', 'x', 'dt', 'A', 'B', 'C', 'D', 'z', 'dt_bias', 'dt_softplus'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

