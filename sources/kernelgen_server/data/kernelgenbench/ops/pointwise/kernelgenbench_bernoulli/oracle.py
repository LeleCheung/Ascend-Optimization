REFERENCE_DEVICE = 'target'

import torch
def _legacy_gen_inputs(ctx, device):
    shape = tuple(ctx["x__shape"]); dtype = getattr(torch, ctx["x__dtype"])
    mode = ctx["mode"]
    result = {"x": torch.rand(shape, dtype=dtype, device=device)}
    if mode == "tensor_p":
        result["p"] = torch.rand(shape, dtype=torch.float32, device=device)
    return result
def run(x, p, mode, generator_seed):
    generator = torch.Generator(device=x.device).manual_seed(generator_seed)
    if mode == "tensor_p":
        return torch.ops.aten.bernoulli.Tensor(x, p, generator=generator)
    if mode == "scalar_p":
        return torch.ops.aten.bernoulli.p(x, float(p), generator=generator)
    return torch.ops.aten.bernoulli(x, generator=generator)
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
