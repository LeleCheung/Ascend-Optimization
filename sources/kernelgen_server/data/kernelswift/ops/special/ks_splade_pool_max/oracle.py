REFERENCE_DEVICE = 'target'

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    seed = int(ctx["__workload_seed"])
    hidden_size = int(ctx["dense_weight__shape"][0])
    vocab_size = int(ctx["decoder_weight__shape"][0])
    _seed_all(seed)
    dense = nn.Linear(hidden_size, hidden_size)
    layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
    decoder = nn.Linear(hidden_size, vocab_size, bias=True)
    values = {
        "seq_lens": torch.tensor(
            [20, 25, 18, 20],
            dtype=getattr(torch, ctx["seq_lens__dtype"]),
            device=device,
        ),
        "dense_weight": dense.weight.detach().to(device),
        "dense_bias": dense.bias.detach().to(device),
        "layer_norm_weight": layer_norm.weight.detach().to(device),
        "layer_norm_bias": layer_norm.bias.detach().to(device),
        "decoder_weight": decoder.weight.detach().to(device),
        "decoder_bias": decoder.bias.detach().to(device),
    }
    _seed_all(seed)
    return values

def run(
    hidden_states,
    seq_lens,
    dense_weight,
    dense_bias,
    layer_norm_weight,
    layer_norm_bias,
    decoder_weight,
    decoder_bias,
):
    x = F.linear(hidden_states, dense_weight, dense_bias)
    x = F.gelu(x)
    x = F.layer_norm(
        x,
        (layer_norm_weight.shape[0],),
        layer_norm_weight,
        layer_norm_bias,
        eps=1e-12,
    )
    x = F.linear(x, decoder_weight, decoder_bias)
    x = torch.log1p(F.relu(x))
    result = []
    offset = 0
    for length in seq_lens.tolist():
        chunk = x[offset:offset + length]
        result.append(chunk.max(dim=0).values)
        offset += length
    return result


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
