REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    num_experts = ctx["b_q_weight__shape"][0]
    K = ctx["size_k"]
    N = ctx["size_n"]
    num_bits = ctx["num_bits"]
    pack_factor = 32 // num_bits
    b_q_weight = torch.randint(0, 2**31, (num_experts, K, N // pack_factor), device=device, dtype=torch.int32)
    perm = torch.arange(K, device=device, dtype=torch.int32).unsqueeze(0).expand(num_experts, -1).contiguous()
    return {"b_q_weight": b_q_weight, "perm": perm}


def run(b_q_weight, perm, size_k, size_n, num_bits):
    return _custom_ops.awq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits, False)


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
