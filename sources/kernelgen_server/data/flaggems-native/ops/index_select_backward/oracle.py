REFERENCE_DEVICE = "target"

import random

import torch


def correctness_run(grad, self_sizes, dim, index):
    # Reproduce pytest reference: float32 scatter-add, cast back to input dtype.
    ref_grad = grad.to(torch.float32)
    ref_out = torch.zeros(self_sizes, dtype=torch.float32, device=grad.device)
    ref_out.index_add_(dim, index.to(torch.int64), ref_grad)
    return ref_out.to(grad.dtype)


def timing_run(grad, self_sizes, dim, index):
    # Benchmark Torch baseline: aten index_select_backward at original dtype.
    return torch.ops.aten.index_select_backward(grad, self_sizes, dim, index)


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    phase = case["phase"]
    dtype = getattr(torch, case["dtype"])

    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(ctx["seed"])
    factory_device = generator.device

    if phase == "correctness":
        test = case["test"]

        if test == "test_index_select_backward":
            shape = tuple(case["shape"])
            py_rng = random.Random(ctx["seed"])
            dim = py_rng.randint(0, len(shape) - 1)
            dim_size_out = shape[dim] + 2
            index_len = shape[dim]
            self_sizes = list(shape)
            self_sizes[dim] = dim_size_out
            self_sizes = tuple(self_sizes)

            grad = torch.randn(shape, dtype=dtype, device=factory_device, generator=generator)
            index = torch.randint(
                0, dim_size_out, (index_len,), device=factory_device, generator=generator
            )
            if factory_device != device:
                grad = grad.to(device)
                index = index.to(device)

            return {
                "grad": grad,
                "self_sizes": self_sizes,
                "dim": dim,
                "index": index,
            }

        # test_index_select_backward_1d: fixed construction
        grad = torch.randn((4,), dtype=dtype, device=factory_device, generator=generator)
        if factory_device != device:
            grad = grad.to(device)
        index = torch.tensor([0, 2, 4, 5], device=device)
        return {
            "grad": grad,
            "self_sizes": (6,),
            "dim": 0,
            "index": index,
        }

    # timing phase
    shape = tuple(case["shape"]["grad"])
    dim_val = case["params"]["dim"]
    self_sizes = tuple(case["params"]["self_sizes"])
    dim_size_out = self_sizes[dim_val]
    index_len = shape[dim_val]

    grad = torch.randn(shape, dtype=dtype, device=factory_device, generator=generator)
    index = torch.randint(
        0, dim_size_out, (index_len,), device=factory_device, generator=generator
    )
    if factory_device != device:
        grad = grad.to(device)
        index = index.to(device)

    return {
        "grad": grad,
        "self_sizes": self_sizes,
        "dim": dim_val,
        "index": index,
    }
