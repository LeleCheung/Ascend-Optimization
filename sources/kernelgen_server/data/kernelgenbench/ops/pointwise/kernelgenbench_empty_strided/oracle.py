REFERENCE_DEVICE = 'target'

import torch

def _current_device():
    for attr, device_type in (("npu", "npu"), ("musa", "musa"), ("mlu", "mlu")):
        runtime = getattr(torch, attr, None)
        if runtime is not None and runtime.is_available():
            return torch.device(device_type, runtime.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")

def run_correctness(size, stride, dtype, layout, pin_memory):
    dtype = getattr(torch, dtype) if dtype else None
    return torch.zeros(tuple(size), dtype=dtype, device=_current_device()).as_strided(size, stride)
def run(size, stride, dtype, layout, pin_memory):
    dtype = getattr(torch, dtype) if dtype else None
    layout = getattr(torch, layout) if layout else None
    return torch.ops.aten.empty_strided(
        size, stride, dtype=dtype, layout=layout,
        pin_memory=pin_memory, device=_current_device()
    )
def _legacy_valid(ref_outputs, sol_outputs, inputs, ctx):
    ref, sol = ref_outputs[0], sol_outputs[0]
    return {"passed": tuple(ref.shape) == tuple(sol.shape) and ref.dtype == sol.dtype
            and tuple(ref.stride()) == tuple(sol.stride()), "message": "metadata mismatch", "metrics": {}}


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
    ordered = [inputs[name] for name in ['size', 'stride', 'dtype', 'layout', 'pin_memory']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )


correctness_run = run_correctness
