REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasStrsm_v2 as _baseline


def _legacy_gen_inputs(ctx, device):
    As = tuple(ctx["A__shape"])
    m = As[0]
    A = torch.randn(m, m, dtype=torch.float32, device=device)
    if m > 0:
        A.diagonal().add_(5.0)
    return {"A": A}


def run(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    _baseline(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    return B


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
