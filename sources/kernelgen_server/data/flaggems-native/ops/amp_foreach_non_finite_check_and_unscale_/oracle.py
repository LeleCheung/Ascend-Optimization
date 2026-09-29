import torch

REFERENCE_DEVICE = "target"


def run(tensors, found_inf, inv_scale):
    """Reference implementation for _amp_foreach_non_finite_check_and_unscale_.

    Mirrors the source tests and benchmark, which exercise the framework op
    ``torch._amp_foreach_non_finite_check_and_unscale_`` directly.  Each tensor
    in ``tensors`` is scaled in-place by ``inv_scale`` (non-finite entries are
    left unchanged) and ``found_inf`` is set to 1.0 if any non-finite value is
    found.
    """
    torch._amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale)
    return None


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    dtype = getattr(torch, case["dtype"])
    mode = case["mode"]

    if mode == "random":
        try:
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError, NotImplementedError):
            generator = None
        if generator is None:
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(ctx["seed"])
            tensors = [
                torch.randn(
                    shape, dtype=dtype, device="cpu", generator=cpu_generator
                ).to(device)
                for shape in case["shapes"]
            ]
        else:
            generator.manual_seed(ctx["seed"])
            tensors = [
                torch.randn(shape, dtype=dtype, device=device, generator=generator)
                for shape in case["shapes"]
            ]
    elif mode == "inf":
        tensors = [
            torch.tensor([1.0, 2.0, float("inf"), 4.0], dtype=dtype, device=device),
            torch.tensor([5.0, 6.0, 7.0], dtype=dtype, device=device),
        ]
    elif mode == "nan":
        tensors = [
            torch.tensor([1.0, 2.0, float("nan"), 4.0], dtype=dtype, device=device),
            torch.tensor([5.0, 6.0, 7.0], dtype=dtype, device=device),
        ]
    else:
        raise ValueError("unknown gen_inputs mode: %r" % (mode,))

    found_inf = torch.tensor(case["found_inf"], dtype=torch.float32, device=device)
    inv_scale = torch.tensor(case["inv_scale"], dtype=torch.float32, device=device)
    return {"tensors": tensors, "found_inf": found_inf, "inv_scale": inv_scale}
