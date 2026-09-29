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
        torch_backend_device = getattr(torch.backends, device_type, None)
        try:
            torch_backend_device.matmul.allow_tf32 = False
        except Exception:
            pass
    return None


def run(input, p, dim, maxnorm):
    return torch.renorm(input, p, dim, maxnorm)
