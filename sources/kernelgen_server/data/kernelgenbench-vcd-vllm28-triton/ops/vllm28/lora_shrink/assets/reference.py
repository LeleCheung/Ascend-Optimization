"""vLLM 参考实现：lora_shrink。LoRA shrink（A 投影）批量分组 GEMM（Triton, in-place）。"""
import torch
try:
    from vllm.lora.ops.triton_ops.lora_shrink_op import lora_shrink as _vllm_lora_shrink
except ModuleNotFoundError:
    _vllm_lora_shrink = None


def _baseline_lora_shrink(
    inputs, lora_a_weights, output_tensor, token_lora_mapping,
    token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc,
    lora_ids, no_lora_flag_cpu, num_active_loras, scaling,
):
    if _vllm_lora_shrink is None:
        raise RuntimeError("vLLM not installed or module not found")
    return _vllm_lora_shrink(
        inputs, lora_a_weights, output_tensor, token_lora_mapping,
        token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc,
        lora_ids, no_lora_flag_cpu, num_active_loras, scaling,
    )


def lora_shrink(*args, **kwargs):
    return _baseline_lora_shrink(*args, **kwargs)
