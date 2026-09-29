"""vLLM 参考实现：gumbel_sample。基于 Gumbel-max 的采样，返回每 token 的采样 id。"""
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample as _vllm_gumbel_sample


def _baseline_gumbel_sample(
    logits,
    expanded_idx_mapping,
    temperature,
    seed,
    pos,
    apply_temperature,
    logits_cache=None,
    logits_cache_col=None,
    use_fp64=False,
):
    return _vllm_gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        logits_cache,
        logits_cache_col,
        use_fp64,
    )


def gumbel_sample(*args, **kwargs):
    return _baseline_gumbel_sample(*args, **kwargs)
