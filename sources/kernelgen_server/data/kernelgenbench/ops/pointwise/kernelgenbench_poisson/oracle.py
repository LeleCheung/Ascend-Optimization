REFERENCE_DEVICE = 'target'

import torch
def _legacy_gen_inputs(ctx, device):
    return {"x": torch.rand(tuple(ctx["x__shape"]),
             dtype=getattr(torch, ctx["x__dtype"]), device=device) + 1.0}
def run(x, use_generator, generator_seed):
    generator = torch.Generator(device=x.device).manual_seed(generator_seed) if use_generator else None
    return torch.ops.aten.poisson(x, generator=generator)
def valid(ref_outputs, sol_outputs, inputs, ctx):
    ref, sol = ref_outputs[0], sol_outputs[0]
    ok = tuple(ref.shape) == tuple(sol.shape) and ref.dtype == sol.dtype
    return {"passed": ok, "message": "随机算子输出契约不一致", "metrics": {}}


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
