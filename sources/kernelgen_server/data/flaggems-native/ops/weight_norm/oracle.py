import torch

REFERENCE_DEVICE = "target"


def timing_run(v, g, dim=0):
    return torch.ops.aten._weight_norm(v, g, dim)


def correctness_run(v, g, dim=0):
    ref_v = v.to(torch.float64)
    ref_g = g.to(torch.float64)
    ref = torch.ops.aten._weight_norm(ref_v, ref_g, dim)
    return ref.to(v.dtype)
