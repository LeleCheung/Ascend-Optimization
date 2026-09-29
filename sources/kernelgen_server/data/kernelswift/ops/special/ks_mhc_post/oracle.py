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
    x = torch.randn(
        tuple(ctx["x__shape"]),
        dtype=getattr(torch, ctx["x__dtype"]),
    )
    residual = torch.randn(
        tuple(ctx["residual__shape"]),
        dtype=getattr(torch, ctx["residual__dtype"]),
    )
    post_layer_mix = torch.randn(
        tuple(ctx["post_layer_mix__shape"]),
        dtype=getattr(torch, ctx["post_layer_mix__dtype"]),
    )
    comb_res_mix = torch.randn(
        tuple(ctx["comb_res_mix__shape"]),
        dtype=getattr(torch, ctx["comb_res_mix__dtype"]),
    )
    discarded_grad = torch.randn(
        tuple(ctx["residual__shape"]),
        dtype=getattr(torch, ctx["residual__dtype"]),
    )
    del discarded_grad
    return {
        "x": x.to(device),
        "residual": residual.to(device),
        "post_layer_mix": post_layer_mix.to(device),
        "comb_res_mix": comb_res_mix.to(device),
    }

def run(x, residual, post_layer_mix, comb_res_mix):
    term2 = torch.einsum(
        "abmn,abmc->abnc",
        comb_res_mix,
        residual.float(),
    )
    return (
        x.float().unsqueeze(-2) * post_layer_mix + term2
    ).bfloat16()


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
