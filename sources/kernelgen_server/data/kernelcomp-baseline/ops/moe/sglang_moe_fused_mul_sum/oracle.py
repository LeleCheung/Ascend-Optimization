REFERENCE_DEVICE = 'target'

import torch
def run(inputs, topk_weights, topk_ids=None, expert_map=None, routed_scaling_factor=None, is_ep=False):
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    w = topk_weights.float() * scale

    if expert_map is not None:
        valid = expert_map[topk_ids.long()] >= 0
        w = w * valid.to(w.dtype)
    elif is_ep:
        valid = topk_ids >= 0
        w = w * valid.to(w.dtype)

    weighted = inputs.float() * w.unsqueeze(-1)
    out = weighted.sum(dim=1)
    return out.to(inputs.dtype)
def _build_case(num_tokens, top_k, size, is_ep=False, use_expert_map=False, num_experts=8,
          routed_scaling_factor=None, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    inputs = torch.randn(
        num_tokens, top_k, size, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    topk_weights = torch.rand(
        num_tokens, top_k, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)

    topk_ids = None
    expert_map = None
    if use_expert_map or is_ep:
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), generator=g, device=_DEVICE, dtype=torch.int32
        )
        if is_ep and not use_expert_map:
            # is_ep (no expert_map): the kernel checks `id_val >= 0` directly,
            # so -1 sentinels in topk_ids are a valid "already dropped" marker.
            drop = torch.rand(num_tokens, top_k, generator=g, device=_DEVICE) < 0.3
            topk_ids = torch.where(drop, torch.full_like(topk_ids, -1), topk_ids)
    if use_expert_map:
        # has_expert_map path indexes `expert_map[topk_ids]` with no id_val>=0
        # guard, so topk_ids must stay valid (dropping is expressed entirely
        # via expert_map's own -1 entries, never via a -1 topk_id).
        expert_map = torch.arange(num_experts, device=_DEVICE, dtype=torch.int32)
        expert_map[num_experts // 2 :] = -1

    return dict(
        inputs=inputs,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        routed_scaling_factor=routed_scaling_factor,
        is_ep=is_ep,
        check=assert_close,
    )

def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = tolerance_for(dtype if dtype is not None else expected.dtype)
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        **tol,
    )

_DEFAULT_TOLERANCE = {"atol": 0.01, "rtol": 0.01}
_TOLERANCES = {"float32": {"atol": 0.0001, "rtol": 0.0001}, "bfloat16": {"atol": 0.015, "rtol": 0.015}, "float16": {"atol": 0.01, "rtol": 0.01}}
def tolerance_for(dtype: torch.dtype) -> dict:
    return dict(_TOLERANCES.get(dtype, _DEFAULT_TOLERANCE))




_DEVICE = None


def _materialize_arg(value, device):
    if isinstance(value, dict) and "__tensor__" in value:
        dtype = value.get("dtype", "float32")
        return torch.tensor(
            value["__tensor__"],
            dtype=getattr(torch, dtype),
            device=device,
        )
    return value


def gen_inputs(ctx, device):
    global _DEVICE
    _DEVICE = device
    case_args = ctx["inputs"].get("_case_args", {})
    args = [_materialize_arg(value, device) for value in case_args.get("args", [])]
    kwargs = {
        key: _materialize_arg(value, device)
        for key, value in case_args.get("kwargs", {}).items()
    }
    dtype = kwargs.get("dtype")
    if isinstance(dtype, str) and dtype:
        kwargs["dtype"] = getattr(torch, dtype)
    constructor = case_args.get("constructor", "_case")
    if constructor == '_case':
        built = _build_case(*args, **kwargs)
    parameters = {'inputs', 'topk_weights', 'topk_ids', 'expert_map', 'routed_scaling_factor', 'is_ep'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

