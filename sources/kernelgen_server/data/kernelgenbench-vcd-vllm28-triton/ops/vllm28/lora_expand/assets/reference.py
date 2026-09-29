"""vLLM 参考实现：lora_expand。LoRA expand（B 投影）批量分组 GEMM（Triton, in-place）。"""
import torch
try:
    from vllm.lora.ops.triton_ops.lora_expand_op import lora_expand as _vllm_lora_expand
except ModuleNotFoundError:
    _vllm_lora_expand = None


def _baseline_lora_expand(
    inputs, lora_b_weights, output_tensor, token_lora_mapping,
    token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc,
    lora_ids, no_lora_flag_cpu, num_active_loras, offset_start=0, add_inputs=False,
):
    if _vllm_lora_expand is None:
        raise RuntimeError("vLLM not installed or module not found")
    return _vllm_lora_expand(
        inputs, lora_b_weights, output_tensor, token_lora_mapping,
        token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc,
        lora_ids, no_lora_flag_cpu, num_active_loras, offset_start, add_inputs,
    )


def lora_expand(*args, **kwargs):
    return _baseline_lora_expand(*args, **kwargs)
