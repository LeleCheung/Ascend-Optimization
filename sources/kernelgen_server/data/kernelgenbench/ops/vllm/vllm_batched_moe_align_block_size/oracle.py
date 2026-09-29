REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    max_tokens = ctx["max_tokens"]
    block_size = ctx["block_size"]
    num_experts = ctx["expert_num_tokens__shape"][0]
    expert_num_tokens = torch.randint(1, max_tokens // 2, (num_experts,), device=device, dtype=torch.int32)
    padded_max = ((max_tokens + block_size - 1) // block_size) * block_size
    sorted_ids_size = num_experts * padded_max
    num_blocks = sorted_ids_size // block_size
    sorted_ids = torch.zeros(sorted_ids_size, device=device, dtype=torch.int32)
    expert_ids = torch.zeros(num_blocks, device=device, dtype=torch.int32)
    num_tokens_post_pad = torch.zeros(1, device=device, dtype=torch.int32)
    return {"expert_num_tokens": expert_num_tokens, "sorted_ids": sorted_ids, "expert_ids": expert_ids, "num_tokens_post_pad": num_tokens_post_pad}


def run(max_tokens, block_size, expert_num_tokens, sorted_ids, expert_ids, num_tokens_post_pad):
    _custom_ops.batched_moe_align_block_size(max_tokens, block_size, expert_num_tokens, sorted_ids, expert_ids, num_tokens_post_pad)
    return (sorted_ids, expert_ids, num_tokens_post_pad)


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
