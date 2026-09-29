REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasCgemm_v2 as _baseline


def _legacy_gen_inputs(ctx, device):
    ash = tuple(ctx["A__shape"])
    bsh = tuple(ctx["B__shape"])
    csh = tuple(ctx["C__shape"])
    A = torch.randn(ash, dtype=torch.float32, device=device) + 1j * torch.randn(ash, dtype=torch.float32, device=device)
    B = torch.randn(bsh, dtype=torch.float32, device=device) + 1j * torch.randn(bsh, dtype=torch.float32, device=device)
    C = torch.randn(csh, dtype=torch.float32, device=device) + 1j * torch.randn(csh, dtype=torch.float32, device=device)
    return {"A": A.to(torch.complex64), "B": B.to(torch.complex64), "C": C.to(torch.complex64)}


def run(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    _baseline(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
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
