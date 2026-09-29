REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def _q8_0(num_experts, row, n_blocks, device):
    scales = (torch.randn(num_experts, row, n_blocks, device=device, dtype=torch.float32).abs() * 0.5 + 0.1).to(torch.float16)
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(num_experts, row, n_blocks, 2)
    qs = torch.randint(0, 256, (num_experts, row, n_blocks, 32), device=device, dtype=torch.uint8)
    return torch.cat([scale_bytes, qs], dim=3).reshape(num_experts, row, n_blocks * 34)


def _legacy_gen_inputs(ctx, device):
    num_experts, row, nbytes = ctx['W__shape']
    n_blocks = nbytes // 34
    W = _q8_0(num_experts, row, n_blocks, device)
    nt = ctx['sorted_token_ids__shape'][0]
    sorted_token_ids = torch.arange(nt, device=device, dtype=torch.int32)
    expert_ids = torch.randint(0, num_experts, (nt,), device=device, dtype=torch.int32)
    num_tokens_post_padded = torch.tensor([nt], device=device, dtype=torch.int32)
    return {'W': W, 'sorted_token_ids': sorted_token_ids, 'expert_ids': expert_ids, 'num_tokens_post_padded': num_tokens_post_padded}


def run(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, quant_type, row, top_k, tokens):
    return _custom_ops.ggml_moe_a8(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, quant_type, row, top_k, tokens)


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
