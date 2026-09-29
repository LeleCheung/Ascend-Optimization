REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    in_shape = tuple(ctx["input__shape"])
    in_dtype = getattr(torch, ctx["input__dtype"])
    qw_shape = tuple(ctx["qweight__shape"])
    sc_shape = tuple(ctx["scales__shape"])
    sc_dtype = getattr(torch, ctx["scales__dtype"])
    qz_shape = tuple(ctx["qzeros__shape"])
    input_tensor = torch.randn(in_shape, device=device, dtype=in_dtype)
    qweight = torch.randint(0, 2**31 - 1, qw_shape, device=device, dtype=torch.int32)
    scales = torch.rand(sc_shape, device=device, dtype=sc_dtype) + 0.01
    qzeros = torch.randint(0, 2**31 - 1, qz_shape, device=device, dtype=torch.int32)
    return {"input": input_tensor, "qweight": qweight, "scales": scales, "qzeros": qzeros}


def run(input, qweight, scales, qzeros, split_k_iters):
    return _custom_ops.awq_gemm(input, qweight, scales, qzeros, split_k_iters)


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
