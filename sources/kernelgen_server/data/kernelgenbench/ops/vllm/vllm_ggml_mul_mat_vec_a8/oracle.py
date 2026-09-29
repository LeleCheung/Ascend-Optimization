REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    w_shape = tuple(ctx["W__shape"])
    block_size = 18
    m = w_shape[0]
    num_qblocks = w_shape[1] // block_size
    W = torch.zeros(m, num_qblocks * block_size, device=device, dtype=torch.uint8)
    for bi in range(num_qblocks):
        off = bi * block_size
        W[:, off] = 0x00
        W[:, off + 1] = 0x3C
        W[:, off + 2:off + 18] = torch.randint(0, 255, (m, 16), device=device, dtype=torch.uint8)
    return {"W": W}


def run(W, X, quant_type, row):
    return _custom_ops.ggml_mul_mat_vec_a8(W, X, quant_type, row)


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
