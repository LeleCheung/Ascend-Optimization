REFERENCE_DEVICE = 'target'

import torch

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
    _seed_all(int(ctx["__workload_seed"]))
    hidden_states = torch.randn(
        tuple(ctx["hidden_states__shape"]),
        dtype=getattr(torch, ctx["hidden_states__dtype"]),
    )
    gating_output = torch.randn(
        tuple(ctx["gating_output__shape"]),
        dtype=getattr(torch, ctx["gating_output__dtype"]),
    )
    return {
        "hidden_states": hidden_states.to(device),
        "gating_output": gating_output.to(device),
    }

def run(hidden_states, gating_output):
    topk = 8
    renormalize = True
    num_expert_group = 8
    topk_group = 4
    routed_scaling_factor = 1.0

    assert hidden_states.size(0) == gating_output.size(0)
    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.size(0)
    experts_per_group = scores.size(-1) // num_expert_group
    group_scores = scores.view(
        num_token, num_expert_group, -1
    ).max(dim=-1).values
    group_idx = torch.topk(
        group_scores, k=topk_group, dim=-1
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group, experts_per_group
    ).reshape(num_token, -1)
    tmp_scores = scores.masked_fill(
        ~score_mask.bool(), float("-inf")
    )
    topk_weights, topk_ids = torch.topk(
        tmp_scores, k=topk, dim=-1
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        )
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return (
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
    )


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
