"""flash-linear-attention 参考实现：fused_recurrent_gsa_fwd。
"""
from fla.ops.gsa import fused_recurrent_gsa


def _baseline_fused_recurrent_gsa_fwd(q, k, v, s, g):
    o, _ = fused_recurrent_gsa(q, k, v, s, g, output_final_state=False)
    return o


def fused_recurrent_gsa_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_gsa_fwd(*args, **kwargs)
