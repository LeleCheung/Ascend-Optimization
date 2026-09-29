REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    M, K = tuple(ctx["a__shape"])
    N = ctx["n"]
    qweight = torch.randint(0, 255, (K, N), device=device, dtype=torch.uint8)
    scale = torch.randn(1, N, device=device, dtype=torch.float16)
    rw, rs, _ = _custom_ops.allspark_repack_weight(qweight, scale)
    return {"b_qweight": rw, "b_scales": rs}


def run(a, b_qweight, b_scales, b_qzeros, n, group_size, sm_count,
        sm_version, CUBLAS_M_THRESHOLD, has_zp, n32k16_reorder):
    return _custom_ops.allspark_w8a16_gemm(
        a, b_qweight, b_scales, b_qzeros, n, group_size, sm_count,
        sm_version, CUBLAS_M_THRESHOLD, has_zp, n32k16_reorder)


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
