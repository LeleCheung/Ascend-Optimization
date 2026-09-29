REFERENCE_DEVICE = 'target'

import torch

def _legacy_gen_inputs(ctx, device):
    num_embeddings = ctx["weight__shape"][0]
    index_shape = tuple(ctx["indices__shape"])
    return {"indices": torch.randint(0, num_embeddings, index_shape, dtype=torch.int64, device=device)}

def run(weight, indices, padding_idx):
    return torch.ops.aten.embedding(weight, indices, padding_idx, False, False)


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
