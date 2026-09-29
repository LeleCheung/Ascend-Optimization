"""flash-linear-attention 参考实现：fused_recurrent_dplr_delta_rule。
"""
from fla.ops import fused_recurrent_dplr_delta_rule as _fla_fused_recurrent_dplr_delta_rule


def _baseline_fused_recurrent_dplr_delta_rule(q, k, v, a, b, gk):
    o, _ = _fla_fused_recurrent_dplr_delta_rule(q, k, v, a, b, gk, output_final_state=False)
    return o


def fused_recurrent_dplr_delta_rule(*args, **kwargs):
    return _baseline_fused_recurrent_dplr_delta_rule(*args, **kwargs)
