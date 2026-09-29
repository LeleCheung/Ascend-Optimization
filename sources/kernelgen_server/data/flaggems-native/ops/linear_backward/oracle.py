import torch
import flag_gems

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


def correctness_run(input, grad_output, weight, output_mask):
    orig_dtype = input.dtype
    input_shape = input.shape
    out_features = weight.shape[0]
    in_features = weight.shape[1]
    batch_dims = input_shape[:-1]
    batch_size = 1
    for d in batch_dims:
        batch_size *= d

    inp_ref = input.to(torch.float64).view(batch_size, in_features)
    go_ref = grad_output.to(torch.float64).view(batch_size, out_features)
    w_ref = weight.to(torch.float64)

    grad_input = None
    grad_weight = None
    grad_bias = None

    if output_mask[0]:
        grad_input = (go_ref @ w_ref).view(*batch_dims, in_features).to(orig_dtype)
    if output_mask[1]:
        grad_weight = (go_ref.t() @ inp_ref).to(orig_dtype)
    if output_mask[2]:
        grad_bias = go_ref.sum(dim=0).to(orig_dtype)

    return grad_input, grad_weight, grad_bias


def timing_run(input, grad_output, weight, output_mask):
    return flag_gems.linear_backward(input, grad_output, weight, output_mask)
