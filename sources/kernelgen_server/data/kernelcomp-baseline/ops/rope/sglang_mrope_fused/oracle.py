REFERENCE_DEVICE = 'target'

import torch
def _rows_for_axis(cos_sin_cache, positions_axis):
    return cos_sin_cache[positions_axis.long()]
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

    t_end = mrope_section[0]
    h_end = t_end + mrope_section[1]

    t_row = _rows_for_axis(cos_sin_cache, positions[0])
    h_row = _rows_for_axis(cos_sin_cache, positions[1])
    w_row = _rows_for_axis(cos_sin_cache, positions[2])

    idx = torch.arange(half_rd, device=q.device)
    t_mask = idx < t_end
    h_mask = (idx >= t_end) & (idx < h_end)
    w_mask = (idx >= h_end) & (idx < half_rd)

    cos = torch.where(t_mask, t_row[:, :half_rd], torch.zeros_like(t_row[:, :half_rd]))
    cos = torch.where(h_mask, h_row[:, :half_rd], cos)
    cos = torch.where(w_mask, w_row[:, :half_rd], cos)

    sin = torch.where(t_mask, t_row[:, half_rd:rotary_dim], torch.zeros_like(cos))
    sin = torch.where(h_mask, h_row[:, half_rd:rotary_dim], sin)
    sin = torch.where(w_mask, w_row[:, half_rd:rotary_dim], sin)

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



