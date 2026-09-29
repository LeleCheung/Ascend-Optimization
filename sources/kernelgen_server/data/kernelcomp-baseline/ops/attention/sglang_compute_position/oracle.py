REFERENCE_DEVICE = 'target'

import torch
def run(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    bs = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == bs
    cumsum = torch.cumsum(extend_seq_lens, dim=0)
    start = torch.zeros_like(cumsum)
    start[1:] = cumsum[:-1]

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    for i in range(bs):
        s = int(extend_seq_lens[i])
        p = int(extend_prefix_lens[i]) if has_prefix else 0
        beg = int(start[i])
        positions[beg : beg + s] = torch.arange(
            p, p + s, dtype=torch.int64, device=extend_seq_lens.device
        )
    return positions, start.to(torch.int32)
def _build_case(bs, max_len=128, with_prefix=True, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    seq_lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=_DEVICE, generator=g
    )
    if with_prefix:
        prefix = torch.randint(
            0, 4096, (bs,), dtype=torch.int32, device=_DEVICE, generator=g
        )
    else:
        prefix = torch.zeros(0, dtype=torch.int32, device=_DEVICE)
    return dict(
        extend_prefix_lens=prefix,
        extend_seq_lens=seq_lens,
        extend_seq_lens_sum=int(seq_lens.sum()),
        check=_check,
    )

def _check(actual, expected):
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])




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
    parameters = {'extend_prefix_lens', 'extend_seq_lens', 'extend_seq_lens_sum'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

