REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    M, K = tuple(ctx["a__shape"])
    a_dtype = getattr(torch, ctx["a__dtype"])
    Kp, N = tuple(ctx["b_q_weight__shape"])
    pack_factor = K // Kp
    bit = 32 // pack_factor
    group_count = tuple(ctx["b_gptq_scales__shape"])[0]
    group_size = K // group_count
    sc_dtype = getattr(torch, ctx["b_gptq_scales__dtype"])

    a_fp = torch.randn(M, K, device=device, dtype=a_dtype)
    b_g_idx = (torch.arange(K, device=device, dtype=torch.int32) // group_size).contiguous()
    shifts = (torch.arange(pack_factor, device=device, dtype=torch.int32) * bit).view(1, 1, -1)
    qvals_w = torch.randint(0, (1 << bit), (Kp, N, pack_factor), device=device, dtype=torch.int32)
    b_q_weight = torch.sum(qvals_w << shifts, dim=-1).to(torch.int32).contiguous()
    gp = (group_count + pack_factor - 1) // pack_factor
    qvals_z = torch.randint(0, (1 << bit), (gp, N, pack_factor), device=device, dtype=torch.int32)
    b_gptq_qzeros = torch.sum(qvals_z << shifts, dim=-1).to(torch.int32).contiguous()
    b_gptq_scales = (torch.rand(group_count, N, device=device, dtype=sc_dtype) + 0.01)
    return {"a": a_fp, "b_q_weight": b_q_weight, "b_gptq_qzeros": b_gptq_qzeros, "b_gptq_scales": b_gptq_scales, "b_g_idx": b_g_idx}


def run(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_exllama, use_v2_format, bit):
    return _custom_ops.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_exllama, use_v2_format, bit)


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
