REFERENCE_DEVICE = 'target'

import torch
def run(
    req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx, kv_indices
):
    out = kv_indices.clone()
    for i in range(req_pool_indices.shape[0]):
        beg = int(kv_indptr[i])
        n = int(page_kernel_lens[i])
        start = int(kv_start_idx[i]) if kv_start_idx is not None else 0
        pool = int(req_pool_indices[i])
        out[beg : beg + n] = req_to_token[pool, start : start + n]
    return out
def _build_case(bs, max_len=2048, max_batch=256, max_context=8192, with_start=False, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    req_to_token = torch.randint(
        0, 1 << 20, (max_batch, max_context), dtype=torch.int32, device=_DEVICE, generator=g
    )
    req_pool_indices = torch.randperm(max_batch, device=_DEVICE, generator=g)[:bs].to(
        torch.int32
    )
    lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=_DEVICE, generator=g
    )
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=_DEVICE)
    kv_indptr[1:] = torch.cumsum(lens, dim=0)
    kv_start = (
        torch.randint(0, 128, (bs,), dtype=torch.int32, device=_DEVICE, generator=g)
        if with_start
        else None
    )
    kv_indices = torch.zeros(int(kv_indptr[-1]), dtype=torch.int32, device=_DEVICE)
    return dict(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        page_kernel_lens=lens,
        kv_indptr=kv_indptr,
        kv_start_idx=kv_start,
        kv_indices=kv_indices,
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
    parameters = {'req_to_token', 'req_pool_indices', 'page_kernel_lens', 'kv_indptr', 'kv_start_idx', 'kv_indices'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

