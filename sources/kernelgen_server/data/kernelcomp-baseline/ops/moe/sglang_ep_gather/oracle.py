REFERENCE_DEVICE = 'target'

import torch
def run(input_tensor, recv_topk_ids, recv_topk_weight, input_index, output_tensor):
    num_tokens, hidden = output_tensor.shape
    topk = recv_topk_ids.shape[1]

    acc = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=input_tensor.device)
    for j in range(topk):
        expert_id = recv_topk_ids[:, j]
        src = input_index[:, j].to(torch.int64)
        w = recv_topk_weight[:, j].to(torch.float32)
        rows = input_tensor[src].to(torch.float32) * w[:, None]
        # Slots whose expert id is negative were never dispatched.
        acc += torch.where((expert_id >= 0)[:, None], rows, torch.zeros_like(rows))

    return acc.to(output_tensor.dtype)
def _build_case(num_tokens, hidden, topk, num_src, dtype=torch.bfloat16, seed=0, drop=0.0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    input_tensor = torch.randn(
        num_src, hidden, dtype=torch.float32, device=_DEVICE, generator=g
    ).to(dtype)
    ids = torch.randint(
        0, 32, (num_tokens, topk), device=_DEVICE, generator=g, dtype=torch.int32
    )
    if drop:
        mask = torch.rand(ids.shape, device=_DEVICE, generator=g) < drop
        ids = torch.where(mask, torch.full_like(ids, -1), ids)
    weight = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=_DEVICE, generator=g
    )
    index = torch.randint(
        0, num_src, (num_tokens, topk), device=_DEVICE, generator=g, dtype=torch.int32
    )
    out = torch.full(
        (num_tokens, hidden), 5.0, dtype=torch.float32, device=_DEVICE
    ).to(dtype)
    return dict(
        input_tensor=input_tensor.contiguous(),
        recv_topk_ids=ids.contiguous(),
        recv_topk_weight=weight.contiguous(),
        input_index=index.contiguous(),
        output_tensor=out.contiguous(),
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
    parameters = {'input_tensor', 'recv_topk_ids', 'recv_topk_weight', 'input_index', 'output_tensor'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

