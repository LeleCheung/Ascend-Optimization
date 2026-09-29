import torch

REFERENCE_DEVICE = "target"


def correctness_run(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    # Reproduce the pytest reference: upcast both operands to float64,
    # run the ATen backward, then cast the result back to the tested dtype
    # (the pytest comparison casts the reference back to the candidate dtype).
    ref_grad_output = grad_output.to(torch.float64)
    ref_input = input.to(torch.float64)
    ref = torch.ops.aten.avg_pool3d_backward(
        ref_grad_output,
        ref_input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    return ref.to(grad_output.dtype)


def timing_run(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    # Reproduce the benchmark's autograd backward baseline directly at the
    # original dtype, without rebuilding or timing the forward graph.
    return torch.ops.aten.avg_pool3d_backward(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
