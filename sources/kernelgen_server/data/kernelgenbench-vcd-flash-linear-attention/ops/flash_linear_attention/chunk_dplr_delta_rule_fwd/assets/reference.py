"""flash-linear-attention 参考实现：chunk_dplr_delta_rule_fwd。
"""
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule


def _baseline_chunk_dplr_delta_rule_fwd(q, k, v, a, b, gk):
    o, _ = chunk_dplr_delta_rule(q, k, v, a, b, gk, output_final_state=False)
    return o


def chunk_dplr_delta_rule_fwd(*args, **kwargs):
    return _baseline_chunk_dplr_delta_rule_fwd(*args, **kwargs)
