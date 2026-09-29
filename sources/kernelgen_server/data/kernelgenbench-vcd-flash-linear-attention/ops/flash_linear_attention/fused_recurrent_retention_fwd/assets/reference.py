"""flash-linear-attention 参考实现：fused_recurrent_retention_fwd。
"""
from fla.ops.retention import fused_recurrent_retention


def _baseline_fused_recurrent_retention_fwd(q, k, v):
    o, _ = fused_recurrent_retention(q, k, v, output_final_state=False)
    return o


def fused_recurrent_retention_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_retention_fwd(*args, **kwargs)
