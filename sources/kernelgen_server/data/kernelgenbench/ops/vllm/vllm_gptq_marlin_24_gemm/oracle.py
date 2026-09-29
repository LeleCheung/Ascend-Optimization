REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import marlin_24_quantize
except ModuleNotFoundError:
    _custom_ops = None
    scalar_types = None

def _legacy_gen_inputs(ctx, device):
    M = ctx['M']
    N = ctx['N']
    K = ctx['K']
    num_bits = ctx['num_bits']
    group_size = ctx['group_size']
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    weight = torch.randn(K, N, device=device, dtype=torch.float16)
    _, b_q_weight, meta, b_scales = marlin_24_quantize(
        weight, scalar_types.uint4b8, group_size)
    workspace = torch.zeros(N, device=device, dtype=torch.int32)
    return {'a': a, 'b_q_weight': b_q_weight, 'meta': meta, 'b_scales': b_scales, 'workspace': workspace}

def run(a, b_q_weight, meta, b_scales, workspace, num_bits, group_size, M, N, K):
    quant_type = scalar_types.uint4b8
    output = _custom_ops.gptq_marlin_24_gemm(a, b_q_weight, meta, b_scales, workspace, quant_type, M, N, K)
    return output

def _legacy_valid(ref_outs, cand_outs, inputs, ctx):
    import torch
    ref_out = ref_outs[0]
    cand_out = cand_outs[0]
    if ref_out.shape != cand_out.shape:
        return {'passed': False, 'message': f'shape mismatch', 'metrics': {}}
    diff = (ref_out.float() - cand_out.float()).abs()
    max_diff = diff.max().item()
    rtol, atol = (0.01, 0.1)
    threshold = atol + rtol * ref_out.abs().float().max().item()
    passed = max_diff <= threshold
    return {'passed': passed, 'message': f'max_diff={max_diff:.6f}', 'metrics': {'max_absolute_error': max_diff}}


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
    ordered = [inputs[name] for name in ['a', 'b_q_weight', 'meta', 'b_scales', 'workspace', 'num_bits', 'group_size', 'M', 'N', 'K']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
