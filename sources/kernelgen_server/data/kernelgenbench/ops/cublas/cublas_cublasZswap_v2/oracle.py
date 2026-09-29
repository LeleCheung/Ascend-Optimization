REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasZswap_v2 as _baseline


def _legacy_gen_inputs(ctx, device):
    xs = tuple(ctx["x__shape"])
    ys = tuple(ctx["y__shape"])
    x = torch.randn(xs, dtype=torch.float64, device=device) + 1j * torch.randn(xs, dtype=torch.float64, device=device)
    y = torch.randn(ys, dtype=torch.float64, device=device) + 1j * torch.randn(ys, dtype=torch.float64, device=device)
    return {"x": x.to(torch.complex128), "y": y.to(torch.complex128)}


def run(n, x, incx, y, incy):
    _baseline(n, x, incx, y, incy)
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
