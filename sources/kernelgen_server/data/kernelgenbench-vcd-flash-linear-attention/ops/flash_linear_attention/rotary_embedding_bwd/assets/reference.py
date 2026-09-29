"""flash-linear-attention 参考实现：rotary_embedding_bwd。

注意：对输入 q/k 直接求梯度，不再依赖 module；cos/sin 作为纯张量随 input 传入。
dq/dk 语义不变。interleaved=False 与 RotaryEmbedding 默认配置一致。
"""
from fla.modules.rotary import rotary_embedding


def _baseline_rotary_embedding_bwd(q, k, dq, dk, cos, sin):
    q = q.clone().detach().requires_grad_()
    k = k.clone().detach().requires_grad_()
    q_rot = rotary_embedding(q, cos, sin, interleaved=False)
    k_rot = rotary_embedding(k, cos, sin, interleaved=False)
    ((q_rot * dq).sum() + (k_rot * dk).sum()).backward()
    return {"dq": q.grad, "dk": k.grad}


def rotary_embedding_bwd(*args, **kwargs):
    return _baseline_rotary_embedding_bwd(*args, **kwargs)
