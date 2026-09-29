import torch

REFERENCE_DEVICE = "target"


def gen_inputs(ctx, device):
    device_type = torch.device(device).type
    if device_type == "musa":
        torch.backends.mudnn.allow_tf32 = False
    elif device_type == "npu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        try:
            torch_backend_device = getattr(torch.backends, device_type)
            torch_backend_device.matmul.allow_tf32 = False
        except Exception:
            pass
    return None


def correctness_run(input, weight, bias=None):
    ref_input = input.to(torch.float64)
    ref_weight = weight.to(torch.float64)
    ref_bias = bias.to(torch.float64) if bias is not None else None
    out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    return out.to(input.dtype)


def timing_run(input, weight, bias=None):
    return torch.nn.functional.linear(input, weight, bias)
