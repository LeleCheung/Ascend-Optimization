REFERENCE_DEVICE = 'target'

import torch
def run(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
    b = mixed_qkvz.shape[0]
    r = num_heads_v // num_heads_qk

    qkvz = mixed_qkvz.reshape(b, num_heads_qk, 2 * head_qk + 2 * r * head_v)
    q = qkvz[:, :, :head_qk]
    k = qkvz[:, :, head_qk : 2 * head_qk]
    v = qkvz[:, :, 2 * head_qk : 2 * head_qk + r * head_v]
    z = qkvz[:, :, 2 * head_qk + r * head_v :]

    # mixed_qkv packs all Q heads, then all K heads, then all V.
    mixed_qkv = torch.cat(
        [q.reshape(b, -1), k.reshape(b, -1), v.reshape(b, -1)], dim=-1
    ).contiguous()

    ba = mixed_ba.reshape(b, num_heads_qk, 2 * r)
    b_out = ba[:, :, :r].reshape(b, num_heads_v).contiguous()
    a_out = ba[:, :, r:].reshape(b, num_heads_v).contiguous()
    return mixed_qkv, z.reshape(b, num_heads_v, head_v).contiguous(), b_out, a_out
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



