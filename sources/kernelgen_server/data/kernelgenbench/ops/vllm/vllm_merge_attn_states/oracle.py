REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    p_lse_shape = tuple(ctx["prefix_lse__shape"])
    s_lse_shape = tuple(ctx["suffix_lse__shape"])
    prefix_lse = (torch.rand(p_lse_shape, device=device, dtype=torch.float32) - 0.5) * 4.0
    suffix_lse = (torch.rand(s_lse_shape, device=device, dtype=torch.float32) - 0.5) * 4.0
    return {"prefix_lse": prefix_lse, "suffix_lse": suffix_lse}


def run(prefix_output, prefix_lse, suffix_output, suffix_lse):
    output = torch.empty_like(prefix_output)
    _custom_ops.merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse, None)
    return output


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
