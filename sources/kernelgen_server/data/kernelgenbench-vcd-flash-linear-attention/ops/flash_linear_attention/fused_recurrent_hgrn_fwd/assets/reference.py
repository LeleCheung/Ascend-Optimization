"""flash-linear-attention 参考实现：fused_recurrent_hgrn_fwd。
"""
from fla.ops.hgrn import fused_recurrent_hgrn


def _baseline_fused_recurrent_hgrn_fwd(x, g):
    o, _ = fused_recurrent_hgrn(x, g, output_final_state=False)
    return o


def fused_recurrent_hgrn_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_hgrn_fwd(*args, **kwargs)
