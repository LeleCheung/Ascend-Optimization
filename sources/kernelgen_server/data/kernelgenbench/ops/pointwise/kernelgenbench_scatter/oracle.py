REFERENCE_DEVICE = 'target'

import torch
def _legacy_gen_inputs(ctx, device):
    x_shape = tuple(ctx["x__shape"])
    index_shape = tuple(ctx["index__shape"])
    src_shape = tuple(ctx["src__shape"])
    x = torch.randn(x_shape, dtype=getattr(torch, ctx["x__dtype"]), device=device)
    src = torch.randn(src_shape, dtype=getattr(torch, ctx["src__dtype"]), device=device)
    dim = int(ctx["dim"])
    index = torch.empty(index_shape, dtype=torch.int64, device=device)
    size_dim = min(src_shape[dim], x_shape[dim])
    m, n, o = index_shape
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                position = [i, j, k]
                position[dim] = slice(0, index_shape[dim])
                index[tuple(position)] = torch.randperm(size_dim, device=device)[:index_shape[dim]]
    return {"x": x, "index": index, "src": src}
def run(x, dim, index, src, reduce):
    if reduce is None:
        return torch.ops.aten.scatter.src(x, dim, index, src)
    return torch.ops.aten.scatter.reduce(x, dim, index, src, reduce=reduce)


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
