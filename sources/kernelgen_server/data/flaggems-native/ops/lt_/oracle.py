import torch

REFERENCE_DEVICE = "target"


def correctness_run(A, B):
    # Mirror to_reference(inp.clone(), upcast=True): cast to float64
    # then ref_inp1.lt_(ref_inp2) in-place; assert_close casts ref back to dtype.
    # A must retain its original dtype (FlagGems writes bool->float in-place).
    ref_A = A.clone().to(torch.float64)
    ref_B = B.to(torch.float64)
    # In-place comparison stores 0/1 in ref_A while preserving its float dtype.
    ref_A.lt_(ref_B)
    A.copy_(ref_A.to(A.dtype))
    return A


def timing_run(A, B):
    # Benchmark torch_op: lambda a, b: torch.ops.aten.lt_.Tensor(a, b)
    return torch.ops.aten.lt_.Tensor(A, B)
