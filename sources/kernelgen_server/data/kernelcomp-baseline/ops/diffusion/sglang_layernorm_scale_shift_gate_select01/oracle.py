REFERENCE_DEVICE = 'target'

import torch
def run(x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index, eps):
    b, l, c = x.shape
    xf = x.to(torch.float32)
    mean = xf.mean(dim=-1, keepdim=True)
    xbar = xf - mean
    var = (xbar * xbar).mean(dim=-1, keepdim=True)
    x_hat = xbar * torch.rsqrt(var + eps)
    if weight is not None:
        x_hat = x_hat * weight.to(torch.float32)
    if bias is not None:
        x_hat = x_hat + bias.to(torch.float32)

    # Per (batch, token), `index` picks modulation set 0 or 1.
    sel = index.to(torch.bool)[..., None]  # [B, L, 1]
    scale = torch.where(sel, scale1[:, None].to(torch.float32), scale0[:, None].to(torch.float32))
    shift = torch.where(sel, shift1[:, None].to(torch.float32), shift0[:, None].to(torch.float32))
    gate = torch.where(sel, gate1[:, None], gate0[:, None]).expand(b, l, c)

    y = x_hat * (1.0 + scale) + shift
    return y.to(x.dtype), gate.to(x.dtype).contiguous()
def _source_check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e)

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



