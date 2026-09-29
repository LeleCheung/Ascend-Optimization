"""flash-linear-attention 参考实现：chunk_gated_delta_rule_fwd。
"""
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def _baseline_chunk_gated_delta_rule_fwd(q, k, v, beta, g):
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=False)
    return o


def chunk_gated_delta_rule_fwd(*args, **kwargs):
    return _baseline_chunk_gated_delta_rule_fwd(*args, **kwargs)
