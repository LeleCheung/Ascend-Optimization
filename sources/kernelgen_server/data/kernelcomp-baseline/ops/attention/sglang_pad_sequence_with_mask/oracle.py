REFERENCE_DEVICE = 'target'

import torch
def run(input_emb, offsets, lengths, max_len):
    b = offsets.shape[0]
    hidden = input_emb.shape[1]
    output = torch.zeros(
        (b, max_len, hidden), device=input_emb.device, dtype=input_emb.dtype
    )
    seq = torch.arange(max_len, device=input_emb.device)
    valid = seq[None, :] < lengths.long()[:, None]
    rows = offsets.long()[:, None] + seq[None, :]
    gathered = input_emb[rows.clamp(min=0, max=input_emb.shape[0] - 1)]
    output = torch.where(valid[..., None], gathered, output)
    return b, output, valid.reshape(-1)
def _build_case(bs, max_len=64, hidden=4096, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    lengths = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=_DEVICE, generator=g
    )
    offsets = torch.zeros(bs, dtype=torch.int32, device=_DEVICE)
    offsets[1:] = torch.cumsum(lengths, dim=0)[:-1]
    total = int(lengths.sum())
    emb = torch.randn(total, hidden, device=_DEVICE, generator=g).to(torch.bfloat16)
    return dict(
        input_emb=emb,
        offsets=offsets,
        lengths=lengths,
        max_len=max_len,
        check=_check,
    )

def _check(actual, expected):
    assert actual[0] == expected[0]
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])




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
    parameters = {'input_emb', 'offsets', 'lengths', 'max_len'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

