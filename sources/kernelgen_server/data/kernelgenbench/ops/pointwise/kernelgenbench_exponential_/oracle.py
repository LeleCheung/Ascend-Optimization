REFERENCE_DEVICE = 'target'

import torch

def run(x, lambd):
    return torch.ops.aten.exponential_(x, lambd)

def _legacy_valid(ref_outputs, sol_outputs, inputs, ctx):
    ref, sol = ref_outputs[0], sol_outputs[0]
    ok = tuple(ref.shape) == tuple(sol.shape) and ref.dtype == sol.dtype and bool(torch.isfinite(sol).all())
    return {"passed": ok, "message": "random operator output contract mismatch", "metrics": {}}


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


VALID_OWNS_RETURN_CONTRACT = True

def valid(ref_outputs, sol_outputs, inputs, ctx):
    ordered = [inputs[name] for name in ['x', 'lambd']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
