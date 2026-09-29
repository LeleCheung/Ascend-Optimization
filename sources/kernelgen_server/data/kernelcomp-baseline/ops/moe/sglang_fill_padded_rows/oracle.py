REFERENCE_DEVICE = 'target'

import torch
def run(x, num_token_non_padded, fill_value):
    out = x.clone()
    n = int(num_token_non_padded)
    if n < out.shape[0]:
        out[n:] = fill_value
    return out
def _build_case(rows, cols, valid, dtype=torch.int32, fill_value=-1, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    if dtype.is_floating_point:
        x = torch.randn(rows, cols, device=_DEVICE, generator=g).to(dtype)
    else:
        x = torch.randint(
            0, 256, (rows, cols), dtype=dtype, device=_DEVICE, generator=g
        )
    return dict(
        x=x,
        num_token_non_padded=torch.tensor([valid], dtype=torch.int32, device=_DEVICE),
        fill_value=fill_value,
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
    parameters = {'x', 'num_token_non_padded', 'fill_value'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

