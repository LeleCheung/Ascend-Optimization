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
    _seed_all(seed)
    w1 = torch.empty(
        tuple(ctx["w1__shape"]),
        dtype=getattr(torch, ctx["w1__dtype"]),
    )
    w2 = torch.empty(
        tuple(ctx["w2__shape"]),
        dtype=getattr(torch, ctx["w2__dtype"]),
    )
    nn.init.normal_(w1, std=0.02)
    nn.init.normal_(w2, std=0.02)
    values = {"w1": w1.to(device), "w2": w2.to(device)}
    # Official get_inputs() is independently replayed from the case seed.
    _seed_all(seed)
    return values

def run(hidden_states, router_logits, w1, w2):
    top_k = 2
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]
    dtype = hidden_states.dtype
    scores = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(
        scores, top_k, dim=-1
    )
    topk_weights = topk_weights / topk_weights.sum(
        dim=-1, keepdim=True
    )
    topk_weights = topk_weights.to(dtype)
    flat_ids = topk_ids.view(-1)
    flat_weights = topk_weights.view(-1)
    x_rep = hidden_states.unsqueeze(1).expand(
        -1, top_k, -1
    ).reshape(-1, hidden_size)
    w1 = w1.to(dtype)
    w2 = w2.to(dtype)
    expert_out = torch.zeros_like(x_rep)
    for expert in range(w1.shape[0]):
        mask = flat_ids == expert
        if not mask.any():
            continue
        x_e = x_rep[mask]
        gate_up = x_e @ w1[expert].T
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_out[mask] = activated @ w2[expert].T
    expert_out = expert_out * flat_weights.unsqueeze(-1)
    return expert_out.view(
        num_tokens, top_k, hidden_size
    ).sum(dim=1)


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
