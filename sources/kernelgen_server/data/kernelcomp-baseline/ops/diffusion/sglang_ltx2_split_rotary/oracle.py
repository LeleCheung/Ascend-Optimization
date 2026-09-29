REFERENCE_DEVICE = 'target'

import torch
def run(x, cos, sin):
    batch, seq_len, inner = x.shape
    _, num_heads, _, half = cos.shape
    head_dim = half * 2
    xv = x.reshape(batch, seq_len, num_heads, head_dim)
    # cos/sin are [B, H, T, half]; align to [B, T, H, half].
    c = cos.permute(0, 2, 1, 3)
    s = sin.permute(0, 2, 1, 3)

    x1, x2 = xv[..., :half], xv[..., half:]
    o1 = (x1 * c).to(torch.bfloat16).float() - x2.float() * s.float()
    o2 = (x2 * c).to(torch.bfloat16).float() + x1.float() * s.float()
    out = torch.cat([o1, o2], dim=-1)
    return out.reshape(batch, seq_len, inner).to(x.dtype)
def _build_case(batch, seq_len, num_heads=16, head_dim=128, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    half = head_dim // 2
    x = torch.randn(
        batch, seq_len, num_heads * head_dim, device=_DEVICE, generator=g
    ).to(torch.bfloat16)
    ang = torch.randn(batch, num_heads, seq_len, half, device=_DEVICE, generator=g)
    return dict(
        x=x,
        cos=torch.cos(ang).to(torch.bfloat16),
        sin=torch.sin(ang).to(torch.bfloat16),
        check=assert_close,
    )

def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = tolerance_for(dtype if dtype is not None else expected.dtype)
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        **tol,
    )

_DEFAULT_TOLERANCE = {"atol": 0.01, "rtol": 0.01}
_TOLERANCES = {"float32": {"atol": 0.0001, "rtol": 0.0001}, "bfloat16": {"atol": 0.015, "rtol": 0.015}, "float16": {"atol": 0.01, "rtol": 0.01}}
def tolerance_for(dtype: torch.dtype) -> dict:
    return dict(_TOLERANCES.get(dtype, _DEFAULT_TOLERANCE))




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
    parameters = {'x', 'cos', 'sin'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

