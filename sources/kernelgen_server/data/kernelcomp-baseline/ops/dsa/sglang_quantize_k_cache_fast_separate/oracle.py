REFERENCE_DEVICE = 'target'

import torch
_FP8_DTYPE = torch.float8_e4m3fn
def run(k_nope, k_rope, group_size=128):
    num_tokens, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]
    num_tiles = dim_nope // group_size
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max

    nope_part_bytes = dim_nope + num_tiles * 4
    rope_part_bytes = dim_rope * k_rope.element_size()

    nope_part_u8 = torch.empty(num_tokens, nope_part_bytes, dtype=torch.uint8, device=k_nope.device)
    rope_part_u8 = torch.empty(num_tokens, rope_part_bytes, dtype=torch.uint8, device=k_rope.device)

    nope_q = nope_part_u8[:, :dim_nope].view(_FP8_DTYPE)
    nope_s = nope_part_u8[:, dim_nope:].view(torch.float32)
    rope_view = rope_part_u8.view(torch.bfloat16)
    rope_view[:] = k_rope

    for tile_idx in range(num_tiles):
        tile = k_nope[:, tile_idx * group_size : (tile_idx + 1) * group_size].float()
        scale = tile.abs().amax(dim=-1) / fp8_max
        nope_s[:, tile_idx] = scale
        quant = (tile / scale.unsqueeze(-1)).clamp(fp8_min, fp8_max).to(_FP8_DTYPE)
        nope_q[:, tile_idx * group_size : (tile_idx + 1) * group_size] = quant

    return nope_part_u8.unsqueeze(1), rope_part_u8.unsqueeze(1)
_FP8_DTYPE = torch.float8_e4m3fn
dim_nope = 512
group_size = 128
def _source_check(actual, expected):
    a_nope_u8, a_rope_u8 = actual
    e_nope_u8, e_rope_u8 = expected

    a_r = a_rope_u8.view(torch.bfloat16)
    e_r = e_rope_u8.view(torch.bfloat16)
    assert torch.equal(a_r, e_r), "rope passthrough bytes must match exactly"

    a_q = a_nope_u8[..., :dim_nope].view(_FP8_DTYPE)
    a_s = a_nope_u8[..., dim_nope:].view(torch.float32)
    e_q = e_nope_u8[..., :dim_nope].view(_FP8_DTYPE)
    e_s = e_nope_u8[..., dim_nope:].view(torch.float32)
    assert_close(a_s, e_s, dtype=torch.float32)

    a_scale_exp = a_s.repeat_interleave(group_size, dim=-1)
    e_scale_exp = e_s.repeat_interleave(group_size, dim=-1)
    a_deq = a_q.float() * a_scale_exp
    e_deq = e_q.float() * e_scale_exp
    atol, rtol = 2e-2, 2e-2
    mismatch = (a_deq - e_deq).abs() > (atol + rtol * e_deq.abs())
    assert mismatch.float().mean() < 2e-2, "too many fp8 rounding-boundary mismatches"

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



