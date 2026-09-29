REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None

def _legacy_gen_inputs(ctx, device):
    num_tokens, hidden_size = ctx['input__shape']
    input_tensor = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    return {'input': input_tensor, 'scale': scale}

def run(input, scale, num_token_padding, use_per_token_if_dynamic):
    return _custom_ops.scaled_fp8_quant(
        input, scale=scale, num_token_padding=num_token_padding,
        use_per_token_if_dynamic=use_per_token_if_dynamic)

def _legacy_valid(ref_outs, cand_outs, inputs, ctx):
    import torch
    ref_output, ref_scale = ref_outs
    cand_output, cand_scale = cand_outs
    if ref_output.shape != cand_output.shape:
        return {'passed': False, 'message': 'output shape mismatch', 'metrics': {}}
    ref_fp16 = ref_output.to(torch.float16)
    cand_fp16 = cand_output.to(torch.float16)
    valid_tokens = ctx['input__shape'][0]
    if ref_output.shape[0] > valid_tokens:
        ref_fp16 = ref_fp16[:valid_tokens]
        cand_fp16 = cand_fp16[:valid_tokens]
    if ref_scale.dim() >= 2 and ref_scale.shape[0] > valid_tokens:
        ref_scale = ref_scale[:valid_tokens]
        cand_scale = cand_scale[:valid_tokens]
    diff = (ref_fp16 - cand_fp16).abs()
    max_diff = diff.max().item()
    scale_diff = (ref_scale - cand_scale).abs().max().item()
    rtol, atol = (0.01, 0.1)
    threshold = atol + rtol * ref_fp16.abs().max().item()
    passed = max_diff <= threshold and scale_diff < 1e-05
    return {'passed': passed, 'message': f'max_diff={max_diff:.6f}, scale_diff={scale_diff:.6e}', 'metrics': {'max_absolute_error': max_diff, 'scale_error': scale_diff}}


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
    ordered = [inputs[name] for name in ['input', 'scale', 'num_token_padding', 'use_per_token_if_dynamic']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
