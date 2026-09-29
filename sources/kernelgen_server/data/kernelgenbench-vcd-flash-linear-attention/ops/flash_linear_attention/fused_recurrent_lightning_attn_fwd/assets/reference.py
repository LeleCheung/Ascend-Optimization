"""flash-linear-attention 参考实现：fused_recurrent_lightning_attn_fwd。
"""
from fla.ops.lightning_attn import fused_recurrent_lightning_attn


def _baseline_fused_recurrent_lightning_attn_fwd(q, k, v, g, layer_idx, num_layers):
    o, _ = fused_recurrent_lightning_attn(q, k, v, layer_idx, num_layers, output_final_state=False)
    return o


def fused_recurrent_lightning_attn_fwd(*args, **kwargs):
    return _baseline_fused_recurrent_lightning_attn_fwd(*args, **kwargs)
