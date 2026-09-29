REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F


def run(hidden_states, w1, w2, topk_output, activation="silu",
        routed_scaling_factor=1.0):
    """SGLang fused_experts semantics, unquantized path.

    ABI follows fused_experts(hidden_states, w1, w2, topk_output,
    moe_runner_config, ...): topk_output is the StandardTopKOutput triple
    (topk_weights, topk_ids, router_logits), unpacked exactly as the source does
    with `topk_weights, topk_ids, _ = topk_output`. The two scalars are the only
    moe_runner_config fields the reference benchmark varies; a dataclass is not
    expressible as a public kernel ABI.

    Arithmetic, from fused_experts_impl plus SiluAndMul/GeluAndMul:

      w1 is [E, N, H] with N = 2 * intermediate. invoke_fused_moe_kernel computes
      hidden @ w1[e].T giving [N], then silu_and_mul halves it. SiluAndMul splits
      as d = x.shape[-1] // 2; silu(x[..., :d]) * x[..., d:], i.e. gate is the
      FIRST half of the w1 rows and up is the second half.
      Then the [N/2] activation goes through w2[e].T giving [H].
      Per-expert results are weighted by topk_weights and summed (moe_sum_reduce),
      and the sum is scaled by routed_scaling_factor.

    Accumulation is done in float32 and cast once at the end, matching the
    kernel's fp32 accumulator with a bf16/fp16 store.

    Not covered, deliberately, because they are pinned off in the benchmarked
    configuration: fp8/int8/int4 quantization and its scales/zero-points,
    b1/b2 bias, no_combine, apply_router_weight_on_input, gemm1_alpha/gemm1_limit
    (the gpt-oss swiglu variant), and inplace.
    """
    topk_weights, topk_ids, _ = topk_output

    num_tokens, H = hidden_states.shape
    E, N, _ = w1.shape
    inter = N // 2
    topk = topk_ids.shape[1]

    if activation == "silu":
        act = F.silu
    elif activation == "gelu":
        act = F.gelu
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    out = torch.zeros(num_tokens, H, dtype=torch.float32, device=hidden_states.device)

    ids = topk_ids.long()
    weights = topk_weights.float()
    x = hidden_states.float()

    # Group tokens by expert so each w1[e]/w2[e] is touched once. Same
    # arithmetic as a per-(token, slot) loop, but it keeps the large timing
    # shapes tractable.
    for e in range(E):
        hit = ids == e
        if not bool(hit.any()):
            continue
        token_idx, slot_idx = torch.nonzero(hit, as_tuple=True)
        xe = x.index_select(0, token_idx)                      # [n, H]
        gate_up = xe @ w1[e].float().transpose(0, 1)           # [n, N]
        gate = gate_up[:, :inter]
        up = gate_up[:, inter:]
        activated = act(gate) * up                             # [n, N/2]
        expert_out = activated @ w2[e].float().transpose(0, 1)  # [n, H]
        scale = weights[token_idx, slot_idx].unsqueeze(-1)
        out.index_add_(0, token_idx, expert_out * scale)

    out = out * float(routed_scaling_factor)
    return out.to(hidden_states.dtype)


def gen_inputs(ctx, device):
    """Materialize MoE inputs with the routing constraints the kernel requires.

    Three reasons a plain random recipe will not do:
      * topk_ids must be valid expert indices in [0, E);
      * topk_weights are normalized routing weights, as produced by
        select_experts, not arbitrary noise;
      * topk_output is a tuple, so it has to be assembled here.

    Weight tensors are filled expert-by-expert: torch.randn on a full
    [E, N, H] bf16 tensor materializes a large fp32 temporary on NPU, which
    OOMs on the E=256 timing shapes. Per-expert filling keeps the distribution
    identical. (Same technique as key_ops_fused_experts_impl.)
    """
    spec = ctx["inputs"]
    dtype = getattr(torch, spec["hidden_states"]["dtype"])
    num_tokens, H = spec["hidden_states"]["shape"]
    E, N, _ = spec["w1"]["shape"]
    # topk_output is a tuple, so its shapes cannot be expressed as a recipe for a
    # parameter; the workload carries them in a context block instead.
    topk = int(spec["topk_output_shapes"]["topk"])

    seed = int(ctx.get("seed", 0))
    torch.manual_seed(seed)

    hidden_states = torch.randn(num_tokens, H, dtype=dtype, device=device)

    w1 = torch.empty(E, N, H, dtype=dtype, device=device)
    for e in range(E):
        w1[e] = torch.randn(N, H, dtype=dtype, device=device)
    w2 = torch.empty(E, H, N // 2, dtype=dtype, device=device)
    for e in range(E):
        w2[e] = torch.randn(H, N // 2, dtype=dtype, device=device)

    # Route the way select_experts does -- softmax over the router logits, then
    # topk. Deriving ids from topk (rather than randint) matters: it guarantees
    # the topk experts of a token are distinct, which real routing always
    # satisfies and which parts of the MoE stack rely on.
    logits = torch.randn(num_tokens, E, dtype=torch.float32, device=device)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_output": (topk_weights, topk_ids, logits),
    }
