import torch

REFERENCE_DEVICE = "target"


def gen_inputs(ctx, device):
    case = ctx["inputs"].get("case")
    if case is None:
        return None

    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"]["input"])
    kind = case["kind"]

    if kind == "large_values":
        inp = torch.full(shape, 1000.0, dtype=dtype, device=device)
    elif kind == "negative_large_values":
        inp = torch.full(shape, -1000.0, dtype=dtype, device=device)
        inp[:, 0] = 1.0
    elif kind == "all_negative_inf":
        inp = torch.full(shape, float("-inf"), dtype=dtype, device=device)
    elif kind == "zeros":
        inp = torch.zeros(shape, dtype=dtype, device=device)
    elif kind == "extreme_mixed":
        inp = torch.zeros(shape, dtype=dtype, device=device)
        inp[:, 0] = 1000.0
        inp[:, 1] = -1000.0
    else:
        raise ValueError("unknown special_logsumexp input kind: " + kind)

    return {"inp": inp, "dim": case["params"]["dim"]}


def correctness_run(inp, dim, keepdim=False):
    # Pytest reference path (to_reference with upcast=True): compute
    # torch.special.logsumexp at float64, then cast back to the tested
    # dtype before returning so the output dtype matches the candidate.
    ref_out = torch.special.logsumexp(
        inp.to(torch.float64), dim=dim, keepdim=keepdim
    )
    return ref_out.to(inp.dtype)


def timing_run(inp, dim, keepdim=False):
    # Benchmark Torch baseline: torch.special.logsumexp at the original dtype.
    return torch.special.logsumexp(inp, dim=dim, keepdim=keepdim)
