"""vLLM 参考实现：count_expert_num_tokens。统计分配到每个本地专家的 token 数。"""
from vllm.model_executor.layers.fused_moe.utils import (
    count_expert_num_tokens as _vllm_count_expert_num_tokens,
)


def _baseline_count_expert_num_tokens(topk_ids, num_local_experts, expert_map):
    return _vllm_count_expert_num_tokens(topk_ids, num_local_experts, expert_map)


def count_expert_num_tokens(*args, **kwargs):
    return _baseline_count_expert_num_tokens(*args, **kwargs)
