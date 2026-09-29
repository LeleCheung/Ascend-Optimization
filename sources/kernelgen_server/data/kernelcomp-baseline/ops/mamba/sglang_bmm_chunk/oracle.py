REFERENCE_DEVICE = 'target'

import math
import torch
def run(a, b, chunk_size, causal=False):
    batch, seqlen, ngroups, k = a.shape
    nchunks = math.ceil(seqlen / chunk_size)

    a_c = a.reshape(batch, nchunks, chunk_size, ngroups, k).float()
    b_c = b.reshape(batch, nchunks, chunk_size, ngroups, k).float()
    out = torch.einsum("bcigk,bcjgk->bcgij", a_c, b_c)
    return out
def _build_case(batch, nchunks, chunk_size, ngroups, k, causal=False, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    seqlen = nchunks * chunk_size
    a = torch.randn(
        batch, seqlen, ngroups, k, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    b = torch.randn(
        batch, seqlen, ngroups, k, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    return dict(a=a, b=b, chunk_size=chunk_size, causal=causal, check=_check_factory(causal, chunk_size))

def _check_factory(causal, chunk_size):
    lower_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=_DEVICE), diagonal=-1
    )

    def _check(actual, expected):
        # Output dtype matches the (bf16) inputs, so compare at bf16
        # tolerance rather than the reference's float32 dtype.
        a = actual.clone()
        e = expected.to(actual.dtype)
        if causal:
            # `causal=True` only guarantees i <= j entries; i > j is arbitrary.
            a[..., lower_mask] = 0
            e[..., lower_mask] = 0
        assert_close(a, e, dtype=actual.dtype)

    return _check




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
    parameters = {'a', 'b', 'chunk_size', 'causal'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

