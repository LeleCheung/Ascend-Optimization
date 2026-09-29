import torch

REFERENCE_DEVICE = "target"


def run(grad_output, self, indices):
    return torch.ops.aten.adaptive_max_pool3d_backward(
        grad_output, self, indices
    )


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    dtype = getattr(torch, case["dtype"])
    shape = tuple(case["shape"]["input"])
    output_size = tuple(case["params"]["output_size"])

    generator = torch.Generator(device="cpu").manual_seed(ctx["seed"])
    x = torch.randn(shape, dtype=dtype, device="cpu", generator=generator).to(device)

    output, indices = torch.nn.functional.adaptive_max_pool3d(
        x, output_size=output_size, return_indices=True
    )
    grad_output = torch.ones_like(output)

    return {
        "grad_output": grad_output,
        "self": x,
        "indices": indices,
    }
