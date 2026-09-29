REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas import cublasCsyrkEx as _baseline


def _legacy_gen_inputs(ctx, device):
    def cx(shape):
        s = tuple(shape)
        return (torch.randn(s, dtype=torch.float32, device=device)
                + 1j * torch.randn(s, dtype=torch.float32, device=device)).to(torch.complex64)
    return {"A": cx(ctx["A__shape"]), "C": cx(ctx["C__shape"])}


def run(uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc):
    _baseline(uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
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
