"""flash-linear-attention 参考实现：fused_recurrent_iplr_delta_rule_fwd。
"""
from fla.ops.generalized_delta_rule.iplr import fused_recurrent_iplr_delta_rule


def _baseline_fused_recurrent_iplr_delta_rule_fwd(q, k, v, a, b):
    o, _ = fused_recurrent_iplr_delta_rule(q, k, v, a, b, output_final_state=False)
    return o


def fused_recurrent_iplr_delta_rule_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_iplr_delta_rule_fwd(*args, **kwargs)
