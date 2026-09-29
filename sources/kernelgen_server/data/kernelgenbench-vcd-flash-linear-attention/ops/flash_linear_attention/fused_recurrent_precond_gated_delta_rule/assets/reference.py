"""flash-linear-attention 参考实现：fused_recurrent_precond_gated_delta_rule。
"""
from fla.ops import fused_recurrent_precond_gated_delta_rule as _fla_fused_recurrent_precond_gated_delta_rule


def _baseline_fused_recurrent_precond_gated_delta_rule(q, k, v, g_atk, g, beta_atk, beta):
    # fla 签名为 (q, k, v, g, gk, gv, beta, g_atk, beta_atk, ...)，用关键字避免位置错配：
    # 测试的 g/beta 为 per-HV 值门，g_atk/beta_atk 为 per-H 键门（ATK 预条件）。
    o, _, _ = _fla_fused_recurrent_precond_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, g_atk=g_atk, beta_atk=beta_atk,
        output_final_state=False)
    return o


def fused_recurrent_precond_gated_delta_rule(*args, **kwargs):
    return _baseline_fused_recurrent_precond_gated_delta_rule(*args, **kwargs)
