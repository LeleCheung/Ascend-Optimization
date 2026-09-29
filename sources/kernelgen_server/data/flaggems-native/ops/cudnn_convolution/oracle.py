import torch

REFERENCE_DEVICE = "target"


def run(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    return torch.cudnn_convolution(
        input,
        weight,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
        benchmark=benchmark,
        deterministic=deterministic,
        allow_tf32=allow_tf32,
    )


def torch_run(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    ndim = input.ndim - 2
    if ndim == 1:
        return torch.conv1d(
            input,
            weight,
            None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    if ndim == 2:
        return torch.conv2d(
            input,
            weight,
            None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    if ndim == 3:
        return torch.conv3d(
            input,
            weight,
            None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    raise ValueError(
        "cudnn_convolution only supports 1D, 2D, and 3D convolutions, "
        f"got input with {ndim} spatial dimensions"
    )
