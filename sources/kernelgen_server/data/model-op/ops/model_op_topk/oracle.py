REFERENCE_DEVICE = 'target'

import torch


def run(topk_weights, topk_ids, gating_output, renormalize=False):
    """SGLang topk_softmax semantics (sgl_kernel.topk_softmax).

    The kernel is in-place over two preallocated buffers, which is the ABI
    sglang/srt/layers/moe/topk.py::fused_topk calls:

        topk_weights = torch.empty(M, topk, dtype=torch.float32, ...)
        topk_ids     = torch.empty(M, topk, dtype=torch.int32,   ...)
        topk_softmax(topk_weights, topk_ids, gating_output, renormalize)

    Arithmetic, per sgl-kernel/tests/test_moe_topk_softmax.py, which asserts the
    kernel against exactly this torch expression:

        softmax_output = torch.softmax(gating_output, dim=-1)
        topk_weights, topk_ids = torch.topk(softmax_output, topk, dim=-1)

    Details that matter:
      * the softmax runs in float32 and the weights stay float32 even when
        gating_output is float16/bfloat16 (the dtype-regression test in that file
        passes fp16/bf16/fp32 gating and always allocates fp32 weights);
      * the output order is torch.topk order, i.e. descending weight -- NOT
        sorted by expert id. This differs from the vLLM topk_softmax operator
        (key_ops_topk_softmax), whose reference re-sorts by expert id because the
        CUDA traversal there is warp-parallel;
      * renormalize divides the selected weights by their sum, matching
        test_topk_softmax_renormalize (which compares fused renormalize against
        the unfused result divided by its own sum).
    """
    k = int(topk_ids.shape[-1])
    probs = torch.softmax(gating_output.float(), dim=-1)
    weights, ids = torch.topk(probs, k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_weights.copy_(weights.to(topk_weights.dtype))
    topk_ids.copy_(ids.to(topk_ids.dtype))
    return None
