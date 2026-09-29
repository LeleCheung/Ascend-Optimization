REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasStrsv_v2 as _baseline


def _legacy_gen_inputs(ctx, device):
    ash = tuple(ctx["A__shape"])
    n = ash[0]
    A = torch.randn(ash, dtype=torch.float32, device=device) + 5.0 * torch.eye(n, dtype=torch.float32, device=device)
    return {"A": A}


def run(uplo, trans, diag, n, A, lda, x, incx):
    _baseline(uplo, trans, diag, n, A, lda, x, incx)
    return x


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
