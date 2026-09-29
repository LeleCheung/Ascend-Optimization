REFERENCE_DEVICE = 'target'

import torch
def run(x, cos, sin, interleaved):
    xf = x.float()
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    c = cos.float().reshape(cos.shape[0], 1, -1)
    s = sin.float().reshape(sin.shape[0], 1, -1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack([o1, o2], dim=-1).reshape(xf.shape)
    return out.to(x.dtype)
def _build_case(tokens, heads=16, head_size=128, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    x = torch.randn(tokens, heads, head_size, device=_DEVICE, generator=g).to(
        torch.bfloat16
    )
    ang = torch.randn(tokens, head_size // 2, device=_DEVICE, generator=g)
    return dict(
        x=x,
        cos=torch.cos(ang).to(torch.bfloat16),
        sin=torch.sin(ang).to(torch.bfloat16),
        interleaved=False,
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
    parameters = {'x', 'cos', 'sin', 'interleaved'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

