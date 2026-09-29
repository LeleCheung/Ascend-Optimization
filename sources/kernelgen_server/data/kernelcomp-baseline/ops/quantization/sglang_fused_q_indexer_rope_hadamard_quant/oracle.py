REFERENCE_DEVICE = 'target'

import torch
_HEAD_DIM = 128
_ROPE_DIM = 64
_FP8_MAX = 448.0
def _hadamard_128(device):
    """Sylvester-order H128: H[i, j] = (-1)^popcount(i & j)."""
    idx = torch.arange(_HEAD_DIM, device=device)
    bits = (idx[:, None] & idx[None, :]).to(torch.int64)
    popcount = torch.zeros_like(bits)
    for b in range(7):
        popcount += (bits >> b) & 1
    return torch.where(
        popcount % 2 == 0,
        torch.ones((), device=device),
        -torch.ones((), device=device),
    )
def run(q_input, weight, weight_scale, freqs_cis, positions):
    b, h, d = q_input.shape
    x = q_input.to(torch.float32)

    # RoPE on the trailing 64 dims, interleaved (real, imag) pairs, one freq
    # row per token shared by every head.
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).to(torch.float32)
    f = freqs_real[positions.to(torch.int64)]  # [B, 64]
    fr, fi = f[:, 0::2][:, None, :], f[:, 1::2][:, None, :]
    rope = x[..., d - _ROPE_DIM :]
    xr, xi = rope[..., 0::2], rope[..., 1::2]
    rotated = torch.stack([xr * fr - xi * fi, xr * fi + xi * fr], dim=-1).flatten(-2)
    y = torch.cat([x[..., : d - _ROPE_DIM], rotated], dim=-1)

    # 128-point Hadamard, normalized by 1/sqrt(128).
    y = (y @ _hadamard_128(x.device)) * (1.0 / (_HEAD_DIM**0.5))

    absmax = y.abs().amax(dim=-1)
    scale = torch.clamp(absmax, min=1e-4) / _FP8_MAX
    q_fp8 = (y / scale[..., None]).to(torch.float8_e4m3fn)
    weights_out = (weight.to(torch.float32) * weight_scale * scale)[..., None]
    return q_fp8, weights_out.to(torch.float32)



def gen_inputs(ctx, device):
    result = {}
    for name, spec in ctx["inputs"].items():
        if not (isinstance(spec, dict) and spec.get("type") == "custom"):
            continue
        result[name] = _build(spec["build"], device)
    return result


def _slice_index(items):
    result = []
    for item in items:
        if isinstance(item, dict) and "start" in item:
            result.append(slice(item["start"], item["stop"], item["step"]))
        elif isinstance(item, dict) and item.get("ellipsis"):
            result.append(Ellipsis)
        else:
            result.append(item)
    return tuple(result)


def _build(node, device):
    op = node["op"]
    if op == "dtype":
        return getattr(torch, node["name"])
    if op == "int":
        return int(_build(node["operand"], device))
    if op == "float":
        return float(_build(node["operand"], device))
    if op in {"randn", "rand"}:
        generator = torch.Generator(device=device).manual_seed(node.get("seed", 0))
        fn = torch.randn if op == "randn" else torch.rand
        return fn(
            tuple(node["shape"]),
            dtype=getattr(torch, node["dtype"]),
            device=device,
            generator=generator,
        )
    if op == "randint":
        generator = torch.Generator(device=device).manual_seed(node.get("seed", 0))
        return torch.randint(
            node["low"],
            node["high"],
            tuple(node["shape"]),
            dtype=getattr(torch, node["dtype"]),
            device=device,
            generator=generator,
        )
    if op == "randperm":
        generator = torch.Generator(device=device).manual_seed(node.get("seed", 0))
        return torch.randperm(
            node["shape"][0],
            dtype=getattr(torch, node["dtype"]),
            device=device,
            generator=generator,
        )
    if op in {"zeros", "ones", "empty"}:
        fn = getattr(torch, op)
        return fn(tuple(node["shape"]), dtype=getattr(torch, node["dtype"]), device=device)
    if op == "full":
        return torch.full(
            tuple(node["shape"]),
            node["fill_value"],
            dtype=getattr(torch, node["dtype"]),
            device=device,
        )
    if op == "tensor_literal":
        return torch.tensor(node["values"], dtype=getattr(torch, node["dtype"]), device=device)
    if op == "to":
        return _build(node["operand"], device).to(getattr(torch, node["dtype"]))
    if op == "reshape":
        return _build(node["operand"], device).reshape(tuple(node["shape"]))
    if op == "transpose":
        return _build(node["operand"], device).transpose(
            *node.get("dims", [0, 1])
        )
    if op == "slice":
        value = _build(node["operand"], device)
        return value[_slice_index(node["index"])]
    if op == "scale":
        return _build(node["operand"], device) * node["value"]
    if op == "shift":
        return _build(node["operand"], device) + node["value"]
    if op == "rshift":
        return node["value"] - _build(node["operand"], device)
    if op == "divide":
        return _build(node["operand"], device) / node["value"]
    if op == "rdivide":
        return node["value"] / _build(node["operand"], device)
    if op == "abs":
        return _build(node["operand"], device).abs()
    if op in {"max_reduce", "min_reduce", "sum_reduce", "mean_reduce", "prod_reduce"}:
        value = _build(node["operand"], device)
        fn = {
            "max_reduce": value.max,
            "min_reduce": value.min,
            "sum_reduce": value.sum,
            "mean_reduce": value.mean,
            "prod_reduce": value.prod,
        }[op]
        dim = node.get("dim")
        if dim is None:
            return fn()
        return fn(dim=dim, keepdim=node.get("keepdim", False))
    if op == "clamp":
        return torch.clamp(
            _build(node["operand"], device),
            min=node.get("min"),
            max=node.get("max"),
        )
    if op == "cat":
        return torch.cat([_build(operand, device) for operand in node["operands"]])
    if op == "stack":
        return torch.stack([_build(operand, device) for operand in node["operands"]])
    if op == "polar":
        return torch.polar(_build(node["operand"], device), _build(node["angle"], device))
    if op == "writes":
        base = _build(node["base"], device)
        for write in node["writes"]:
            value = (
                _build(write["value"], device)
                if isinstance(write["value"], dict)
                else write["value"]
            )
            base[_slice_index(write["index"])] = value
        return base
    if op in {"expand", "permute", "flip", "roll", "unsqueeze", "squeeze", "narrow", "repeat", "logical_not"}:
        value = _build(node["operand"], device)
        if op == "expand":
            return value.expand(tuple(node["shape"]))
        if op == "permute":
            return value.permute(tuple(node["dims"]))
        if op == "flip":
            return value.flip(tuple(node["dims"]))
        if op == "roll":
            return value.roll(node["shifts"], dims=node.get("dims"))
        if op == "unsqueeze":
            return value.unsqueeze(node["dim"])
        if op == "squeeze":
            return value.squeeze(node.get("dim"))
        if op == "narrow":
            return value.narrow(node["dim"], node["start"], node["length"])
        if op == "repeat":
            return value.repeat(tuple(node["sizes"]))
        return value.logical_not()
    if op.startswith("compare_"):
        left = _build(node["operand"], device)
        right = _build(node["rhs"], device) if "rhs" in node else node["value"]
        return {
            "compare_gt": left.__gt__,
            "compare_ge": left.__ge__,
            "compare_lt": left.__lt__,
            "compare_le": left.__le__,
            "compare_eq": left.eq,
            "compare_ne": left.ne,
        }[op](right)
    if op in {"logical_and", "logical_or"}:
        left = _build(node["operand"], device)
        right = _build(node["rhs"], device) if "rhs" in node else node["value"]
        return (left & right) if op == "logical_and" else (left | right)
    if op == "mask_index":
        return _build(node["operand"], device)[_build(node["mask"], device)]
    if op == "masked_fill":
        return _build(node["operand"], device).masked_fill(
            _build(node["mask"], device), node["value"]
        )
    if op == "where":
        return torch.where(
            _build(node["cond"], device),
            _build(node["x"], device) if isinstance(node["x"], dict) else node["x"],
            _build(node["y"], device) if isinstance(node["y"], dict) else node["y"],
        )
    if op in {"eye", "linspace"}:
        dtype = getattr(torch, node["dtype"])
        if op == "eye":
            return torch.eye(node["shape"][0], node["shape"][1], dtype=dtype, device=device)
        return torch.linspace(
            node["start"], node["end"], node["shape"][0], dtype=dtype, device=device
        )
    if op == "arange":
        return torch.arange(
            node["start"],
            node["stop"],
            step=node.get("step", 1),
            dtype=getattr(torch, node.get("dtype", "int64")),
            device=device,
        )
    raise ValueError(f"unsupported build op: {op!r}")

def _source_check(actual, expected):
    a_q, a_w = actual
    e_q, e_w = expected
    # The kernel's Hadamard is a warp butterfly and the reference is a matmul,
    # so the fp32 summation orders differ by ulps before the fp8 rounding.
    assert_close(a_w, e_w, dtype=torch.float32, atol=1e-4, rtol=1e-4)
    a_deq = a_q.to(torch.float32) * a_w
    e_deq = e_q.to(torch.float32) * e_w
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert mismatch.float().mean() < 1e-2, "too many fp8 rounding-boundary mismatches"

def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = tolerance_for(dtype if dtype is not None else expected.dtype)
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        **tol,
    )
def tolerance_for(dtype: torch.dtype) -> dict:
    return dict(_TOLERANCES.get(dtype, _DEFAULT_TOLERANCE))
_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)



def valid(ref_outputs, sol_outputs, inputs, ctx):
    try:
        if len(ref_outputs) > 1:
            _source_check(tuple(ref_outputs), tuple(sol_outputs))
        else:
            _source_check(ref_outputs[0], sol_outputs[0])
    except AssertionError as exc:
        return {"passed": False, "message": str(exc), "metrics": {}}
    return {"passed": True, "message": "", "metrics": {}}



