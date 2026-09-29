REFERENCE_DEVICE = 'target'

import torch

def run(x, other):
    return torch.ops.aten.mul(x, other)

def _legacy_valid(ref_outputs, sol_outputs, inputs, ctx):
    ref, sol = ref_outputs[0], sol_outputs[0]
    if isinstance(ref, torch.Tensor) and ref.ndim == 0 and not isinstance(sol, torch.Tensor):
        sol = torch.as_tensor(sol, dtype=ref.dtype, device=ref.device)
    if not isinstance(ref, torch.Tensor) or not isinstance(sol, torch.Tensor):
        return {"passed": ref == sol, "message": "scalar value mismatch", "metrics": {}}
    if ref.shape != sol.shape or ref.dtype != sol.dtype:
        return {"passed": False, "message": "shape or dtype mismatch", "metrics": {}}
    rtol = {torch.float16: 1e-3, torch.bfloat16: 0.016, torch.float32: 1e-5, torch.float64: 1e-5}.get(ref.dtype, 0)
    try:
        torch.testing.assert_close(sol, ref, rtol=rtol, atol=1e-4 if ref.is_floating_point() else 0)
    except AssertionError as exc:
        return {"passed": False, "message": str(exc), "metrics": {}}
    return {"passed": True, "message": "", "metrics": {}}


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
    ordered = [inputs[name] for name in ['x', 'other']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
