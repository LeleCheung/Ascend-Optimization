REFERENCE_DEVICE = 'target'

import torch


def run(input_tensor):
    """SwiGLU over the last dimension, FlagGems-vllm Ascend semantics.

    The candidate kernel (FlagGems-vllm PR #728) splits the last dimension in
    half and computes silu(a) * b:

        input_2d = input_tensor.contiguous().view(M, 2 * H)
        input_a, input_b = torch.split(input_2d, H, dim=1)
        out = silu(input_a) * input_b

    Precision note: the kernel computes the sigmoid in float32, and the golden it
    is validated against is `torch_npu.npu_swiglu(input_tensor, dim=-1)`. The
    vendor golden keeps the whole expression in float32 and rounds once at the
    end rather than rounding silu(a) back to the input dtype before multiplying.
    Cross-checked on npu:0 over the core_shapes.yaml shapes: this fp32-throughout
    form is bit-identical to npu_swiglu for bfloat16 and float32, and within one
    ulp for float16, whereas rounding silu(a) early is off by up to 6e-2 in
    bfloat16. So the fp32 form is used here.
    """
    if input_tensor.shape[-1] % 2 != 0:
        raise ValueError(
            "The last dimension must be an even number, got "
            f"{input_tensor.shape[-1]}"
        )
    shape = input_tensor.shape
    hidden = shape[-1] // 2
    rows = input_tensor.numel() // (2 * hidden)
    input_2d = input_tensor.contiguous().view(rows, 2 * hidden)
    input_a, input_b = torch.split(input_2d, hidden, dim=1)
    a32 = input_a.to(torch.float32)
    out = (a32 * torch.sigmoid(a32)) * input_b.to(torch.float32)
    return out.to(input_tensor.dtype).view(*shape[:-1], hidden)
