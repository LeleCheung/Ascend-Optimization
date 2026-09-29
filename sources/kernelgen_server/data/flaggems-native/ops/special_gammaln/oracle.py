import torch

REFERENCE_DEVICE = "target"


def run(A):
    return torch.special.gammaln(A)


def gen_inputs(ctx, device):
    inputs = ctx["inputs"]
    if "case" not in inputs:
        return None
    case = inputs["case"]
    if case.get("kind") != "edge_cases":
        return None
    dtype = getattr(torch, case["dtype"])
    return {"A": torch.tensor(case["vals"], dtype=dtype, device=device)}
