"""flash-linear-attention 参考实现：chunk_precond_gated_delta_rule_fwd。
"""
from fla.ops.precond_gated_delta_rule import chunk_precond_gated_delta_rule


def _baseline_chunk_precond_gated_delta_rule_fwd(q, k, v, g_atk, g, beta_atk, beta):
    o, _, _ = chunk_precond_gated_delta_rule(q, k, v, g_atk, g, beta_atk, beta,
                                             output_final_state=False)
    return o


def chunk_precond_gated_delta_rule_fwd(*args, **kwargs):
    return _baseline_chunk_precond_gated_delta_rule_fwd(*args, **kwargs)
