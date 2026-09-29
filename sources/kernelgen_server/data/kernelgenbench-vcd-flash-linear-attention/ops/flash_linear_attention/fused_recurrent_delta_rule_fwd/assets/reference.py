"""flash-linear-attention 参考实现：fused_recurrent_delta_rule_fwd。
"""
from fla.ops.delta_rule import fused_recurrent_delta_rule


def _baseline_fused_recurrent_delta_rule_fwd(q, k, v, beta):
    o, _ = fused_recurrent_delta_rule(q, k, v, beta, output_final_state=False)
    return o


def fused_recurrent_delta_rule_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_delta_rule_fwd(*args, **kwargs)
