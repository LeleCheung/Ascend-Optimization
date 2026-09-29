"""vLLM 参考实现：SamplingMaskTensors。位打包采样 mask（Triton kernel via from_logits）。

SYMBOL 是 NamedTuple 类；其 Triton kernel 经 classmethod from_logits 触发。
导出函数 SamplingMaskTensors(logits, num_sampled_tokens) 调用 from_logits 并返回
(packed_mask, counts) 供比较。
"""
from vllm.v1.worker.gpu.sample.output import (
    SamplingMaskTensors as _SamplingMaskTensors,
)


def _baseline_SamplingMaskTensors(logits, num_sampled_tokens):
    out = _SamplingMaskTensors.from_logits(logits, num_sampled_tokens)
    return out.packed_mask, out.counts


def SamplingMaskTensors(*args, **kwargs):
    return _baseline_SamplingMaskTensors(*args, **kwargs)
