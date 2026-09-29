REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None

def _legacy_gen_inputs(ctx, device):
    num_tokens = ctx['topk_ids__shape'][0]
    topk = ctx['topk_ids__shape'][1]
    num_experts = ctx['num_experts']
    block_size = ctx['block_size']
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_num_tokens_padded // block_size, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    return {'topk_ids': topk_ids, 'sorted_ids': sorted_ids, 'expert_ids': expert_ids, 'num_tokens_post_pad': num_tokens_post_pad}

def run(topk_ids, sorted_ids, expert_ids, num_tokens_post_pad, num_experts, block_size):
    _custom_ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad, None)
    return (sorted_ids, expert_ids, num_tokens_post_pad)

def _legacy_valid(ref_outs, cand_outs, inputs, ctx):
    import torch
    ref_sorted_ids, ref_expert_ids, ref_num_tokens = ref_outs
    cand_sorted_ids, cand_expert_ids, cand_num_tokens = cand_outs
    if not torch.equal(ref_num_tokens, cand_num_tokens):
        return {'passed': False, 'message': 'num_tokens_post_pad mismatch', 'metrics': {}}
    if not torch.equal(ref_expert_ids, cand_expert_ids):
        return {'passed': False, 'message': 'expert_ids mismatch', 'metrics': {}}
    topk_ids = inputs[0]
    num_experts = ctx['num_experts']
    block_size = ctx['block_size']

    def _ceil_div(a, b):
        return (a + b - 1) // b
    start = 0
    for eid in range(num_experts):
        cnt = (topk_ids == eid).sum().item()
        aligned = _ceil_div(cnt, block_size) * block_size
        end = start + aligned
        ref_set = set(ref_sorted_ids[start:end].tolist())
        cand_set = set(cand_sorted_ids[start:end].tolist())
        if ref_set != cand_set:
            return {'passed': False, 'message': f'Expert {eid}: token sets differ', 'metrics': {}}
        start = end
    return {'passed': True, 'message': 'all checks passed', 'metrics': {}}


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


VALID_OWNS_RETURN_CONTRACT = True

def valid(ref_outputs, sol_outputs, inputs, ctx):
    ordered = [inputs[name] for name in ['topk_ids', 'sorted_ids', 'expert_ids', 'num_tokens_post_pad', 'num_experts', 'block_size']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
