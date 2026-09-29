REFERENCE_DEVICE = 'target'

import torch
def run(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    out_c, lse_c = out.clone(), lse.clone()
    zero = (kv_lens == 0).nonzero().flatten().tolist()
    for i in zero:
        beg, end = int(cum_seq_lens[i]), int(cum_seq_lens[i + 1])
        out_c[beg:end] = 0
        lse_c[beg:end] = float("-inf")
    return out_c, lse_c
def _build_case(bs, num_heads=16, v_head_dim=128, max_q=4, zero_frac=0.3, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    q_lens = torch.randint(
        1, max_q + 1, (bs,), dtype=torch.int32, device=_DEVICE, generator=g
    )
    cum = torch.zeros(bs + 1, dtype=torch.int32, device=_DEVICE)
    cum[1:] = torch.cumsum(q_lens, dim=0)
    total = int(cum[-1])
    kv_lens = torch.randint(
        1, 4096, (bs,), dtype=torch.int32, device=_DEVICE, generator=g
    )
    mask = torch.rand(bs, device=_DEVICE, generator=g) < zero_frac
    kv_lens[mask] = 0
    out = torch.randn(
        total, num_heads, v_head_dim, device=_DEVICE, generator=g
    ).to(torch.bfloat16)
    lse = torch.randn(total, num_heads, device=_DEVICE, generator=g)
    return dict(
        out=out,
        lse=lse,
        kv_lens=kv_lens,
        cum_seq_lens=cum,
        max_seq_len=int(q_lens.max()),
        check=_check,
    )

def _check(actual, expected):
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], equal_nan=True)




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
    parameters = {'out', 'lse', 'kv_lens', 'cum_seq_lens', 'max_seq_len'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

