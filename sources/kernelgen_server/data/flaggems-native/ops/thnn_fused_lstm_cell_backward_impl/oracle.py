import torch

REFERENCE_DEVICE = "target"


def gen_inputs(ctx, device):
    case = ctx["inputs"]["case"]
    dtype = getattr(torch, case["dtype"])
    batch_size, hidden_size = case["shape"]
    has_bias = case["params"]["has_bias"]

    try:
        gen = torch.Generator(device=device)
        factory_device = device
    except Exception:
        gen = torch.Generator(device="cpu")
        factory_device = "cpu"
    gen.manual_seed(ctx["seed"])

    input_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=factory_device, generator=gen
    )
    hidden_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=factory_device, generator=gen
    )
    cx = torch.randn(
        batch_size, hidden_size, dtype=dtype, device=factory_device, generator=gen
    )
    input_bias = torch.zeros(4 * hidden_size, dtype=dtype, device=factory_device)
    hidden_bias = torch.randn(
        4 * hidden_size, dtype=dtype, device=factory_device, generator=gen
    )
    grad_hy = torch.randn(
        batch_size, hidden_size, dtype=dtype, device=factory_device, generator=gen
    )
    grad_cy = torch.randn(
        batch_size, hidden_size, dtype=dtype, device=factory_device, generator=gen
    )

    if factory_device == "cpu":
        input_gates = input_gates.to(device)
        hidden_gates = hidden_gates.to(device)
        cx = cx.to(device)
        input_bias = input_bias.to(device)
        hidden_bias = hidden_bias.to(device)
        grad_hy = grad_hy.to(device)
        grad_cy = grad_cy.to(device)

    hx, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )

    return {
        "grad_hy": grad_hy,
        "grad_cy": grad_cy,
        "cx": cx,
        "cy": cy,
        "workspace": workspace,
        "has_bias": has_bias,
    }


def run(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    return torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, has_bias
    )
