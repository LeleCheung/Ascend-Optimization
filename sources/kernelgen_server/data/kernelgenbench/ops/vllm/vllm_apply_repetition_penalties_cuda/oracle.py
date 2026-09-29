REFERENCE_DEVICE = 'target'

import torch
try:
    from vllm import _custom_ops as ops
except ModuleNotFoundError:
    ops = None


def run(logits, prompt_mask, output_mask, repetition_penalties):
    if not isinstance(repetition_penalties, torch.Tensor):
        repetition_penalties = torch.full((logits.shape[0],), float(repetition_penalties), device=logits.device, dtype=logits.dtype)
    ops.apply_repetition_penalties_cuda(logits, prompt_mask, output_mask, repetition_penalties)
    return logits
