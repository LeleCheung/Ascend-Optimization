REFERENCE_DEVICE = 'target'

import torch
def run(
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    tok = input_.reshape(-1).long()
    in_org = (tok >= org_vocab_start_index) & (tok < org_vocab_end_index)
    in_added = (tok >= added_vocab_start_index) & (tok < added_vocab_end_index)
    valid = in_org | in_added

    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    local_id = torch.where(in_org, tok - org_vocab_start_index, tok - added_offset)
    local_id = torch.where(valid, local_id, torch.zeros_like(local_id))

    out = weight[local_id]
    out = out * valid[:, None].to(out.dtype)
    return out.reshape(*input_.shape, weight.shape[1])
def _build_case(num_tokens, hidden=4096, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    org_start, org_end = 1000, 5000
    padding = 8
    added_start, added_end = 20000, 20064
    local_vocab = (org_end - org_start) + padding + (added_end - added_start)
    weight = torch.randn(local_vocab, hidden, device=_DEVICE, generator=g).to(
        torch.bfloat16
    )
    # A mix of in-window, added-window and out-of-window tokens.
    tokens = torch.randint(
        0, 30000, (num_tokens,), dtype=torch.int32, device=_DEVICE, generator=g
    )
    tokens[::3] = torch.randint(
        org_start, org_end, (tokens[::3].numel(),), dtype=torch.int32, device=_DEVICE,
        generator=g,
    )
    tokens[1::3] = torch.randint(
        added_start, added_end, (tokens[1::3].numel(),), dtype=torch.int32,
        device=_DEVICE, generator=g,
    )
    return dict(
        input_=tokens.contiguous(),
        weight=weight,
        org_vocab_start_index=org_start,
        org_vocab_end_index=org_end,
        num_org_vocab_padding=padding,
        added_vocab_start_index=added_start,
        added_vocab_end_index=added_end,
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
    parameters = {'input_', 'weight', 'org_vocab_start_index', 'org_vocab_end_index', 'num_org_vocab_padding', 'added_vocab_start_index', 'added_vocab_end_index'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

