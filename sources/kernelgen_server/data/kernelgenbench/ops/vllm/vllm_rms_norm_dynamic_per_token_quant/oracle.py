REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None

def _legacy_gen_inputs(ctx, device):
    num_tokens, hidden_size = ctx['input__shape']
    input_tensor = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float16, device=device)
    scale = torch.empty(1, dtype=torch.float32, device=device)
    return {'input': input_tensor, 'weight': weight, 'scale': scale}

def run(input, weight, scale, epsilon):
    return _custom_ops.rms_norm_dynamic_per_token_quant(
        input, weight, epsilon, torch.float8_e4m3fn)

def _legacy_valid(ref_outs, cand_outs, inputs, ctx):
    import torch
    ref_output, ref_scales = ref_outs
    cand_output, cand_scales = cand_outs
    if ref_output.shape != cand_output.shape:
        return {'passed': False, 'message': 'output shape mismatch', 'metrics': {}}
    ref_fp16 = ref_output.to(torch.float16)
    cand_fp16 = cand_output.to(torch.float16)
    diff_output = (ref_fp16 - cand_fp16).abs()
    max_diff_output = diff_output.max().item()
    if ref_scales.shape != cand_scales.shape:
        return {'passed': False, 'message': 'scales shape mismatch', 'metrics': {}}
    diff_scales = (ref_scales - cand_scales).abs()
    max_diff_scales = diff_scales.max().item()
    rtol, atol = (0.01, 0.1)
    threshold_output = atol + rtol * ref_fp16.abs().max().item()
    threshold_scales = atol + rtol * ref_scales.abs().max().item()
    passed = max_diff_output <= threshold_output and max_diff_scales <= threshold_scales
    return {'passed': passed, 'message': f'output_diff={max_diff_output:.6f}, scales_diff={max_diff_scales:.6f}', 'metrics': {'max_output_error': max_diff_output, 'max_scales_error': max_diff_scales}}


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
    ordered = [inputs[name] for name in ['input', 'weight', 'scale', 'epsilon']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
