REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F
def run(
    scores,
    bias,
    topk,
    scoring_func="sigmoid",
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
    moe_softcapping=0.0,
    num_expert_group=1,
    topk_group=1,
):
    scores = scores.float()
    bias = bias.float()
    M, N = scores.shape
    K = topk
    K_routed = topk - num_fused_shared_experts
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    if scoring_func == "sigmoid":
        activated = torch.sigmoid(scores)
        biased = activated + bias[None, :]
    elif scoring_func == "sqrtsoftplus":
        activated = torch.sqrt(F.softplus(scores))
        biased = activated + bias[None, :]
    else:
        logit = scores
        if moe_softcapping != 0.0:
            logit = moe_softcapping * torch.tanh(logit / moe_softcapping)
        biased = logit + bias[None, :]
        activated = torch.softmax(biased, dim=-1)

    if num_expert_group > 1:
        experts_per_group = N // num_expert_group
        biased_g = biased.view(M, num_expert_group, experts_per_group)
        top2 = torch.topk(biased_g, 2, dim=-1).values
        group_score = top2.sum(dim=-1)
        keep_idx = torch.topk(group_score, topk_group, dim=-1).indices
        keep_mask_g = torch.zeros(M, num_expert_group, dtype=torch.bool, device=scores.device)
        keep_mask_g.scatter_(1, keep_idx, True)
        keep_mask = (
            keep_mask_g.unsqueeze(-1)
            .expand(M, num_expert_group, experts_per_group)
            .reshape(M, N)
        )
        biased = torch.where(keep_mask, biased, torch.full_like(biased, -float("inf")))

    _, top_idx = torch.topk(biased, K_routed, dim=-1)
    selected_vals = torch.gather(activated, 1, top_idx)
    routed_sum = selected_vals.sum(dim=-1, keepdim=True)

    weights = torch.zeros(M, K, dtype=torch.float32, device=scores.device)
    indices = torch.zeros(M, K, dtype=torch.int32, device=scores.device)
    weights[:, :K_routed] = selected_vals
    indices[:, :K_routed] = top_idx.to(torch.int32)

    num_shared = K - K_routed
    if num_shared > 0:
        shared_weight = routed_sum / routed_scaling_factor
        shared_idx = N + torch.arange(num_shared, device=scores.device, dtype=torch.int32)
        weights[:, K_routed:] = shared_weight.expand(M, num_shared)
        indices[:, K_routed:] = shared_idx[None, :].expand(M, num_shared)

    if renormalize:
        norm = torch.where(routed_sum > 0, routed_sum, torch.ones_like(routed_sum))
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor

    return weights, indices
def _source_check(actual, expected):
    aw, ai = actual
    ew, ei = expected
    assert_close(aw, ew, dtype=torch.float32)
    assert torch.equal(ai, ei), "expert index mismatch"

def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = tolerance_for(dtype if dtype is not None else expected.dtype)
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        **tol,
    )
def tolerance_for(dtype: torch.dtype) -> dict:
    return dict(_TOLERANCES.get(dtype, _DEFAULT_TOLERANCE))
_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)



def valid(ref_outputs, sol_outputs, inputs, ctx):
    try:
        if len(ref_outputs) > 1:
            _source_check(tuple(ref_outputs), tuple(sol_outputs))
        else:
            _source_check(ref_outputs[0], sol_outputs[0])
    except AssertionError as exc:
        return {"passed": False, "message": str(exc), "metrics": {}}
    return {"passed": True, "message": "", "metrics": {}}



