REFERENCE_DEVICE = 'target'

import torch
def run(vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask):
    out = vocab_mask.clone()
    if reset_vocab_mask:
        out[batch_idx] = 0 if is_allowed else -1
    row = out[batch_idx]
    for tid in token_ids:
        word, bit = tid // 32, tid % 32
        if is_allowed:
            row[word] |= 1 << bit
        else:
            row[word] &= ~(1 << bit)
    return out
def _build_case(batch, vocab, num_tokens, is_allowed=True, reset=True, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    words = (vocab + 31) // 32
    mask = torch.randint(
        -(2**31), 2**31 - 1, (batch, words), dtype=torch.int32, device=_DEVICE,
        generator=g,
    )
    ids = torch.randperm(vocab, generator=torch.Generator().manual_seed(seed))[
        :num_tokens
    ].tolist()
    return dict(
        vocab_mask=mask,
        token_ids=ids,
        batch_idx=batch // 2,
        is_allowed=is_allowed,
        reset_vocab_mask=reset,
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
    parameters = {'vocab_mask', 'token_ids', 'batch_idx', 'is_allowed', 'reset_vocab_mask'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

