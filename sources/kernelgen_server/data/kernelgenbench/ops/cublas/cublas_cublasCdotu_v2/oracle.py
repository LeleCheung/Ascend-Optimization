REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasCdotu_v2 as _baseline


def _legacy_gen_inputs(ctx, device):
    xs = tuple(ctx["x__shape"])
    ys = tuple(ctx["y__shape"])
    rs = tuple(ctx["result__shape"])
    x = torch.randn(xs, dtype=torch.float32, device=device) + 1j * torch.randn(xs, dtype=torch.float32, device=device)
    y = torch.randn(ys, dtype=torch.float32, device=device) + 1j * torch.randn(ys, dtype=torch.float32, device=device)
    result = torch.zeros(rs, dtype=torch.complex64, device=device)
    return {"x": x.to(torch.complex64), "y": y.to(torch.complex64), "result": result}


def run(n, x, incx, y, incy, result):
    _baseline(n, x, incx, y, incy, result)
    return result


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
