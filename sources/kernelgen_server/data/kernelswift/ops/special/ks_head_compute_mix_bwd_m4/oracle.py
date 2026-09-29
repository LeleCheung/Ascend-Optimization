REFERENCE_DEVICE = 'target'

import torch

def _seed_all(seed):
    torch.manual_seed(seed)
    for name in ("gcu", "cuda", "npu", "mlu"):
        module = getattr(torch, name, None)
        if module is None:
            continue
        try:
            if module.is_available():
                module.manual_seed_all(seed)
        except Exception:
            pass

def _legacy_gen_inputs(ctx, device):
    _seed_all(int(ctx["__workload_seed"]))
    values = {}
    for name in ("input_mix", "mhc_scale", "mhc_base", "grad_out"):
        values[name] = torch.randn(
            tuple(ctx[f"{name}__shape"]),
            dtype=getattr(torch, ctx[f"{name}__dtype"]),
        )
    return {name: value.to(device) for name, value in values.items()}

def run(input_mix, mhc_scale, mhc_base, grad_out):
    z = input_mix * mhc_scale + mhc_base
    sigmoid = torch.sigmoid(z)
    grad_z = grad_out * sigmoid * (1 - sigmoid)
    grad_input_mix = grad_z * mhc_scale
    grad_mhc_base = grad_z.sum(
        dim=(0, 1), keepdim=True
    ).view(-1)
    grad_mhc_scale = (grad_z * input_mix).sum(
        dim=(0, 1, 2), keepdim=True
    ).view(1)
    return grad_input_mix, grad_mhc_scale, grad_mhc_base


def _legacy_context(ctx):
    result = {"__workload_seed": ctx["seed"]}
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
