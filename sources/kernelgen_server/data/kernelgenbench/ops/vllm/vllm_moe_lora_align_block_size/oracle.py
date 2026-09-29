REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _legacy_gen_inputs(ctx, device):
    topk_ids_shape = tuple(ctx["topk_ids__shape"])
    num_tokens = topk_ids_shape[0]
    topk = topk_ids_shape[1]
    num_experts = ctx["num_experts"]
    topk_ids = torch.randint(0, num_experts, topk_ids_shape, device=device, dtype=torch.int32)
    return {"topk_ids": topk_ids}


def run(topk_ids, num_experts, block_size, max_loras, num_tokens, topk):
    max_num_tokens_padded = num_tokens * topk + num_experts * block_size
    max_num_m_blocks = max_num_tokens_padded // block_size
    token_lora_mapping = torch.zeros(num_tokens, device=topk_ids.device, dtype=torch.int32)
    sorted_token_ids = torch.zeros(max_num_tokens_padded, device=topk_ids.device, dtype=torch.int32)
    experts_ids = torch.zeros(max_num_m_blocks, device=topk_ids.device, dtype=torch.int32)
    num_tokens_post_pad = torch.zeros(1, device=topk_ids.device, dtype=torch.int32)
    adapter_enabled = torch.zeros(max_num_m_blocks, device=topk_ids.device, dtype=torch.int32)
    lora_ids = torch.zeros(max_num_m_blocks, device=topk_ids.device, dtype=torch.int32)
    _custom_ops.moe_lora_align_block_size(
        topk_ids, token_lora_mapping, num_experts, block_size,
        max_loras, max_num_tokens_padded, max_num_m_blocks,
        sorted_token_ids, experts_ids, num_tokens_post_pad,
        adapter_enabled, lora_ids, None
    )
    return (sorted_token_ids, experts_ids, num_tokens_post_pad, adapter_enabled, lora_ids)


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
