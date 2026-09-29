"""vLLM 参考实现：compute_token_logprobs。在指定 token_ids 处计算 log-softmax 概率。"""
from vllm.v1.worker.gpu.sample.logprob import (
    compute_token_logprobs as _vllm_compute_token_logprobs,
)


def _baseline_compute_token_logprobs(logits, token_ids):
    return _vllm_compute_token_logprobs(logits, token_ids)


def compute_token_logprobs(*args, **kwargs):
    return _baseline_compute_token_logprobs(*args, **kwargs)
