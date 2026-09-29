REFERENCE_DEVICE = 'target'

import torch
def run(dst, src, dst_indices_raw, step_indices_raw):
    out = dst.clone()
    valid = step_indices_raw >= 0
    if not bool(valid.any()):
        return out
    req = torch.nonzero(valid, as_tuple=True)[0]
    dst_idx = dst_indices_raw[req].to(torch.int64)
    step_idx = step_indices_raw[req].to(torch.int64)
    # dst[:, dst_indices[i]] = src[:, i, step_indices[i]] for every valid i
    out[:, dst_idx] = src[:, req, step_idx]
    return out
def _build_case(layers, cache, requests, draft, state_shape, invalid=0.0, seed=0, dtype=torch.float32):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    dst = torch.randn(
        layers, cache, *state_shape, dtype=torch.float32, device=_DEVICE, generator=g
    ).to(dtype)
    src = torch.randn(
        layers, requests, draft, *state_shape, dtype=torch.float32, device=_DEVICE, generator=g
    ).to(dtype)
    dst_idx = torch.randperm(cache, generator=g, device=_DEVICE)[:requests].to(torch.int32)
    step = torch.randint(
        0, draft, (requests,), device=_DEVICE, generator=g, dtype=torch.int32
    )
    if invalid:
        mask = torch.rand(step.shape, device=_DEVICE, generator=g) < invalid
        step = torch.where(mask, torch.full_like(step, -1), step)
    return dict(
        dst=dst.contiguous(),
        src=src.contiguous(),
        dst_indices_raw=dst_idx,
        step_indices_raw=step,
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
    parameters = {'dst', 'src', 'dst_indices_raw', 'step_indices_raw'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

