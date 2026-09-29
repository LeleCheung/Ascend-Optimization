"""flash-linear-attention 参考实现：fused_recurrent_precond_kda。
"""
from fla.ops import fused_recurrent_precond_kda as _fla_fused_recurrent_precond_kda


def _baseline_fused_recurrent_precond_kda(q, k, v, g, g_atk, beta_atk, beta):
    o, _, _ = _fla_fused_recurrent_precond_kda(q, k, v, g, g_atk, beta_atk, beta, output_final_state=False)
    return o


def fused_recurrent_precond_kda(*args, **kwargs):
    return _baseline_fused_recurrent_precond_kda(*args, **kwargs)
