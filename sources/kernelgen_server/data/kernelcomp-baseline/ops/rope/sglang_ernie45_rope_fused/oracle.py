REFERENCE_DEVICE = 'target'

import torch
def _apply_rope(x, n_h, head_size, rotary_dim, cos, sin):
    num_tokens = x.shape[0]
    half_rd = rotary_dim // 2
    x = x.view(num_tokens, n_h, head_size).clone()
    x1 = x[..., :half_rd].float()
    x2 = x[..., half_rd:rotary_dim].float()
    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    new1 = x1 * cos_e - x2 * sin_e
    new2 = x2 * cos_e + x1 * sin_e
    out = torch.cat([new1.to(x.dtype), new2.to(x.dtype), x[..., rotary_dim:]], dim=-1)
    return out.view(num_tokens, n_h * head_size)
def run(q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim):
    num_tokens, n_q_dim = q.shape
    n_k_dim = k.shape[1]
    n_qh = n_q_dim // head_size
    n_kh = n_k_dim // head_size
    half_rd = rotary_dim // 2

    section_h, section_w, section_t = mrope_section
    assert section_h == section_w, "Ernie4.5 layout assumes section_h == section_w"
    section_hw = section_h + section_w

    tpos = positions[0].long()
    hpos = positions[1].long()
    wpos = positions[2].long()

    ridx = torch.arange(half_rd, device=q.device)
    use_hw = (ridx < section_hw).unsqueeze(0)
    use_h = ((ridx % 2) == 0).unsqueeze(0)

    pos_hw = torch.where(use_h, hpos.unsqueeze(1), wpos.unsqueeze(1))
    pos = torch.where(use_hw, pos_hw, tpos.unsqueeze(1))  # (T, half_rd)

    col = ridx.unsqueeze(0).expand(num_tokens, half_rd)
    cos = cos_sin_cache[pos, col].float()
    sin = cos_sin_cache[pos, col + half_rd].float()

    q_out = _apply_rope(q, n_qh, head_size, rotary_dim, cos, sin)
    k_out = _apply_rope(k, n_kh, head_size, rotary_dim, cos, sin)
    return q_out, k_out
def _source_check(actual, expected):
    aq, ak = actual
    eq, ek = expected
    assert_close(aq, eq)
    assert_close(ak, ek)

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



