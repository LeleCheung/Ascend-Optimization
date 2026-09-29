REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F
def run(x, gamma, bias, rms_scale, eps):
    c = x.shape[1]
    scale = rms_scale if rms_scale is not None else c**0.5
    shape = (1, c, 1, 1, 1)
    y = F.normalize(x.float(), dim=1, eps=eps) * scale * gamma.float().reshape(shape)
    if bias is not None:
        y = y + bias.float().reshape(shape)
    out_dtype = torch.promote_types(x.dtype, gamma.dtype)
    return F.silu(y).to(out_dtype)
def _build_case(b, c, t, h, w, with_bias=False, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    x = (
        torch.randn(b, c, t, h, w, device=_DEVICE, generator=g)
        .to(torch.bfloat16)
        .contiguous(memory_format=torch.channels_last_3d)
    )
    gamma = torch.randn(c, device=_DEVICE, generator=g).to(torch.bfloat16)
    bias = (
        torch.randn(c, device=_DEVICE, generator=g).to(torch.bfloat16)
        if with_bias
        else None
    )
    return dict(
        x=x, gamma=gamma, bias=bias, rms_scale=None, eps=1e-12, check=assert_close
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
    parameters = {'x', 'gamma', 'bias', 'rms_scale', 'eps'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

