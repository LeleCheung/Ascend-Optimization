"""vLLM 参考实现：vocab_parallel_embedding。对输入 token id 施加词表并行掩码与偏移（get_masked_input_and_mask）。CUDA 算子。"""
from vllm.model_executor.layers.vocab_parallel_embedding import (
    get_masked_input_and_mask as _vllm_compiled,
)

# get_masked_input_and_mask is wrapped with @torch.compile; use the underlying
# eager function to avoid an inductor compile in the sandbox.
_vllm_vocab_parallel_embedding = getattr(_vllm_compiled, "__wrapped__", _vllm_compiled)


def _baseline_vocab_parallel_embedding(
    input_,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    return _vllm_vocab_parallel_embedding(
        input_,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )


def vocab_parallel_embedding(*args, **kwargs):
    return _baseline_vocab_parallel_embedding(*args, **kwargs)
