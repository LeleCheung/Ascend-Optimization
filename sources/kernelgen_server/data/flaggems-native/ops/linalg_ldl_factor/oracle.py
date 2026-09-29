import torch

REFERENCE_DEVICE = "target"


def run(self, *, hermitian=False):
    return torch.linalg.ldl_factor(self, hermitian=hermitian)


def gen_inputs(ctx, device):
    # Reproduce benchmark base.py module-level TF32 precision setup.
    device_type = torch.device(device).type
    if device_type == "musa":
        torch.backends.mudnn.allow_tf32 = False
    elif device_type == "npu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        try:
            getattr(torch.backends, device_type).matmul.allow_tf32 = False
        except Exception:
            pass

    case = ctx["inputs"]["case"]
    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"])
    n = shape[0]

    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(ctx["seed"])
    factory_device = generator.device

    A_raw = torch.randn(shape, dtype=dtype, device=factory_device, generator=generator)
    if factory_device != torch.device(device):
        A_raw = A_raw.to(device)

    eye = torch.eye(n, dtype=dtype, device=device)
    A = A_raw @ A_raw.transpose(-2, -1) + eye * n

    return {"self": A}
