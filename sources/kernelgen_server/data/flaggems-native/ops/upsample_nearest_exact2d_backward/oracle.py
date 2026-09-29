import torch

REFERENCE_DEVICE = "target"


def run(grad_output, output_size, input_size, scales_h=None, scales_w=None):
    if scales_h is None and scales_w is None:
        return torch.ops.aten._upsample_nearest_exact2d_backward.default(
            grad_output, output_size, input_size
        )
    return torch.ops.aten._upsample_nearest_exact2d_backward.default(
        grad_output, output_size, input_size, scales_h, scales_w
    )


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    phase = case["phase"]
    dtype = getattr(torch, case["dtype"])
    if phase == "correctness":
        shape = tuple(case["shape"])
        out_h = shape[2] * 2
        out_w = shape[3] * 2
        output_size = (out_h, out_w)
        grad_output = torch.ones(
            (shape[0], shape[1], out_h, out_w), dtype=dtype, device=device
        )
        inputs = {
            "grad_output": grad_output,
            "output_size": output_size,
            "input_size": tuple(shape),
        }
        if case.get("with_scales"):
            inputs["scales_h"] = 2.0
            inputs["scales_w"] = 2.0
        return inputs
    if phase == "timing":
        grad_output = torch.ones(
            tuple(case["grad_output_shape"]), dtype=dtype, device=device
        )
        return {
            "grad_output": grad_output,
            "output_size": tuple(case["output_size"]),
            "input_size": tuple(case["input_size"]),
        }
    return None
