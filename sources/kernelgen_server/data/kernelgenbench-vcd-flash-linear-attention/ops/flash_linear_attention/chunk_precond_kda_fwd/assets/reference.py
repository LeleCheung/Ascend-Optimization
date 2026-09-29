"""flash-linear-attention 参考实现：chunk_precond_kda_fwd。
"""
from fla.ops.precond_kda import chunk_precond_kda


def _baseline_chunk_precond_kda_fwd(q, k, v, g, g_atk, beta_atk, beta):
    o, _, _ = chunk_precond_kda(q, k, v, g, g_atk, beta_atk, beta, output_final_state=False)
    return o


def chunk_precond_kda_fwd(*args, **kwargs):
    return _baseline_chunk_precond_kda_fwd(*args, **kwargs)
