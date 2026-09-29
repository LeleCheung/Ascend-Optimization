REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None

def _legacy_gen_inputs(ctx, device):
    batch, dim, seqlen = ctx['u__shape']
    dstate = ctx['A__shape'][1]
    u = torch.randn(batch, dim, seqlen, dtype=torch.float16, device=device)
    delta = torch.rand(batch, dim, seqlen, dtype=torch.float16, device=device) * 0.1
    A = -torch.rand(dim, dstate, dtype=torch.float32, device=device) - 0.1
    B = torch.randn(batch, dstate, seqlen, dtype=torch.float16, device=device)
    C = torch.randn(batch, dstate, seqlen, dtype=torch.float16, device=device)
    D = torch.randn(dim, dtype=torch.float32, device=device)
    z_ = torch.randn(batch, dim, seqlen, dtype=torch.float16, device=device)
    delta_bias = torch.zeros(dim, dtype=torch.float32, device=device)
    return {'u': u, 'delta': delta.clone(), 'A': A, 'B': B, 'C': C, 'D': D, 'z_': z_.clone(), 'delta_bias': delta_bias}

def run(u, delta, A, B, C, D, z_, delta_bias, delta_softplus):
    batch, dim, _ = u.shape
    ssm_states = torch.zeros((batch, dim, A.shape[1]), device=u.device, dtype=torch.float32)
    _custom_ops.selective_scan_fwd(
        u, delta, A, B.unsqueeze(1), C.unsqueeze(1), D, z_, delta_bias,
        delta_softplus, None, None, None, ssm_states, -1)
    return ssm_states

def _legacy_valid(ref_outs, cand_outs, inputs, ctx):
    import torch
    ref_states = ref_outs[0]
    cand_states = cand_outs[0]
    if ref_states.shape != cand_states.shape:
        return {'passed': False, 'message': 'ssm_states shape mismatch', 'metrics': {}}
    diff = (ref_states - cand_states).abs()
    max_diff = diff.max().item()
    rtol, atol = (0.01, 0.01)
    threshold = atol + rtol * ref_states.abs().max().item()
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
    ordered = [inputs[name] for name in ['u', 'delta', 'A', 'B', 'C', 'D', 'z_', 'delta_bias', 'delta_softplus']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
