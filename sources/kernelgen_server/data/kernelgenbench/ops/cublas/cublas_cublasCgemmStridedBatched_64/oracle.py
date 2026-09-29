REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasCgemmStridedBatched_64 as _baseline


def _legacy_gen_inputs(ctx, device):
    As = tuple(ctx["A__shape"])
    Bs = tuple(ctx["B__shape"])
    Cs = tuple(ctx["C__shape"])
    A = torch.randn(As, dtype=torch.float32, device=device) + 1j * torch.randn(As, dtype=torch.float32, device=device)
    B = torch.randn(Bs, dtype=torch.float32, device=device) + 1j * torch.randn(Bs, dtype=torch.float32, device=device)
    C = torch.randn(Cs, dtype=torch.float32, device=device) + 1j * torch.randn(Cs, dtype=torch.float32, device=device)
    return {"A": A.to(torch.complex64), "B": B.to(torch.complex64), "C": C.to(torch.complex64)}


def run(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    _baseline(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    return C


def _legacy_context(ctx):
    result = {}
    for name, spec in ctx["inputs"].items():
        kind = spec.get("type") if isinstance(spec, dict) else None
        if kind in {"random", "custom"}:
            result[f"{name}__shape"] = spec["shape"]
            result[f"{name}__dtype"] = spec["dtype"]
            for key, value in spec.items():
                if key not in {"type", "shape", "dtype"}:
                    result[f"{name}__{key}"] = value
        elif kind in {"scalar", "literal"}:
            result[name] = spec["value"]
        else:
            raise ValueError(f"unsupported legacy input recipe: {name}")
    return result


def gen_inputs(ctx, device):
    return _legacy_gen_inputs(_legacy_context(ctx), device)
