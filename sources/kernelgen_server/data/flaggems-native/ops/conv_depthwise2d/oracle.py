import torch

REFERENCE_DEVICE = "target"


def gen_inputs(ctx, device):
    """Reproduce the precision-flag setup performed by the accuracy pytest and
    the benchmark test before the measured callable is invoked."""
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    return None


def run(input, weight, kernel_size, bias, stride, padding, dilation):
    return torch.ops.aten._conv_depthwise2d(
        input, weight, kernel_size, bias, stride, padding, dilation
    )
