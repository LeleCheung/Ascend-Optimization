"""flash-linear-attention 参考实现：fused_recurrent_linear_attn_fwd。
"""
from fla.ops.linear_attn import fused_recurrent_linear_attn


def _baseline_fused_recurrent_linear_attn_fwd(q, k, v):
    o, _ = fused_recurrent_linear_attn(q, k, v, normalize=False, output_final_state=False)
    return o


def fused_recurrent_linear_attn_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_linear_attn_fwd(*args, **kwargs)
