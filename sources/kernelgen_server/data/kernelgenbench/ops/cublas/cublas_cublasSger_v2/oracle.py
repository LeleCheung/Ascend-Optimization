REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasSger_v2 as _baseline


def _legacy_gen_inputs(ctx, device):
    m, n = tuple(ctx["A__shape"])
    dtype = getattr(torch, ctx["A__dtype"] )
    return {"A": torch.randn(n, m, dtype=dtype, device=device).t()}


def run(m, n, alpha, x, incx, y, incy, A, lda):
    _baseline(m, n, alpha, x, incx, y, incy, A, lda)
    return A


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
