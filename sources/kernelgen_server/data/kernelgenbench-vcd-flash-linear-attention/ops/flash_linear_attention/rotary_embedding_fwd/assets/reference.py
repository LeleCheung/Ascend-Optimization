"""flash-linear-attention 参考实现：rotary_embedding_fwd。

注意：RotaryEmbedding 无可训练参数，forward 内部先按 dim/base 确定性地算出 cos/sin
缓存（fp32 计算后 .to(q.dtype)），再对 q/k 调用底层 rotary_embedding 函数。故本 baseline
直接调用 rotary_embedding，cos/sin 作为纯张量随 input 传入（见 accuracy 侧 input_build），
input 可跨机传输。interleaved=False 与 RotaryEmbedding 默认配置一致。
"""
from fla.modules.rotary import rotary_embedding


def _baseline_rotary_embedding_fwd(q, k, cos, sin):
    q_rot = rotary_embedding(q, cos, sin, interleaved=False)
    k_rot = rotary_embedding(k, cos, sin, interleaved=False)
    return q_rot, k_rot


def rotary_embedding_fwd(*args, **kwargs):
    return _baseline_rotary_embedding_fwd(*args, **kwargs)
