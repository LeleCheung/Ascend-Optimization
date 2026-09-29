REFERENCE_DEVICE = 'target'

import torch
def run(
    req_pool_indices, req_to_token, start_offset, batch_size, draft_token_num, device
):
    offsets = torch.arange(draft_token_num, device=device)
    cols = start_offset.long()[:, None] + offsets[None, :]
    rows = req_pool_indices.long()[:, None].expand(-1, draft_token_num)
    return req_to_token[rows, cols].reshape(-1).to(torch.int64)
def _build_case(bs, draft_token_num=8, max_batch=4096, pool_len=8192, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    req_to_token = torch.randint(
        0, 1 << 20, (max_batch, pool_len), dtype=torch.int32, device=_DEVICE, generator=g
    )
    pool_idx = torch.randperm(max_batch, device=_DEVICE, generator=g)[:bs].to(
        torch.int32
    )
    start_offset = torch.randint(
        0, pool_len - draft_token_num, (bs,), dtype=torch.int32, device=_DEVICE,
        generator=g,
    )
    return dict(
        req_pool_indices=pool_idx,
        req_to_token=req_to_token,
        start_offset=start_offset,
        batch_size=bs,
        draft_token_num=draft_token_num,
        device=_DEVICE,
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
    parameters = {'req_pool_indices', 'req_to_token', 'start_offset', 'batch_size', 'draft_token_num', 'device'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

