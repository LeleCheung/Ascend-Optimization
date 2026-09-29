"""flash-linear-attention 参考实现：fused_recurrent_gla_fwd。
"""
from fla.ops.gla import fused_recurrent_gla


def _baseline_fused_recurrent_gla_fwd(q, k, v, g):
    # fused_recurrent_gla 的门控参数名为 gk
    o, _ = fused_recurrent_gla(q, k, v, gk=g, output_final_state=False)
    return o


def fused_recurrent_gla_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_gla_fwd(*args, **kwargs)
