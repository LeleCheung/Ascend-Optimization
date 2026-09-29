REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasCgemmStridedBatched as _baseline


def _legacy_gen_inputs(ctx, device):
    def cx(shape):
        s = tuple(shape)
        return (torch.randn(s, dtype=torch.float32, device=device)
                + 1j * torch.randn(s, dtype=torch.float32, device=device)).to(torch.complex64)
    return {"A": cx(ctx["A__shape"]), "B": cx(ctx["B__shape"]), "C": cx(ctx["C__shape"])}


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
