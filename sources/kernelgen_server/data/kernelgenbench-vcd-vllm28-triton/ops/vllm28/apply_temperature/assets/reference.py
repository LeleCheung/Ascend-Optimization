"""vLLM 参考实现：apply_temperature。对 logits 施加温度缩放（in-place 写回 logits）。"""
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature as _vllm_apply_temperature


def _baseline_apply_temperature(logits, expanded_idx_mapping, temperature):
    _vllm_apply_temperature(logits, expanded_idx_mapping, temperature)


def apply_temperature(*args, **kwargs):
    return _baseline_apply_temperature(*args, **kwargs)
