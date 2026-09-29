REFERENCE_DEVICE = 'target'

import torch
def run(accept_index, out_cache_loc, accept_out_cache_loc, size):
    out = accept_out_cache_loc.clone()
    keep = accept_index[:size] > -1
    n = int(keep.sum())
    if n:
        out[:n] = out_cache_loc[accept_index[:size][keep].long()]
    return out
def _build_case(size, accept_frac=0.5, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    accept_index = torch.arange(size, dtype=torch.int64, device=_DEVICE)
    reject = torch.rand(size, device=_DEVICE, generator=g) > accept_frac
    accept_index[reject] = -1
    out_cache_loc = torch.randint(
        1, 1 << 20, (size,), dtype=torch.int64, device=_DEVICE, generator=g
    )
    return dict(
        accept_index=accept_index,
        out_cache_loc=out_cache_loc,
        accept_out_cache_loc=torch.zeros(size, dtype=torch.int64, device=_DEVICE),
        size=size,
        check=_check,
    )

def _check(actual, expected):
    # Entries past the accepted count are undefined in the kernel's output.
    n = int((expected != 0).sum()) if expected.numel() else 0
    torch.testing.assert_close(actual[: max(n, 1)], expected[: max(n, 1)])




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
    parameters = {'accept_index', 'out_cache_loc', 'accept_out_cache_loc', 'size'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

