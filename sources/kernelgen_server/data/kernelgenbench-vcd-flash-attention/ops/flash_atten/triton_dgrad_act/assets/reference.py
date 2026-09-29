# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

try:
    from flash_attn.ops.triton import linear as _m
except (ModuleNotFoundError, ImportError):
    _m = None


def triton_dgrad_act(grad_output, weight, activation="id", act_input=None):
    """FlashAttention Triton baseline for triton_dgrad_act.

    Fused data-gradient + activation-gradient for a linear+activation layer:
        grad_input = act_grad(act_input) * (grad_output @ weight)
    (for activation="id" the activation gradient is 1, i.e. grad_output @ weight).

    Args:
        grad_output (Tensor): upstream gradient, shape (..., out_features).
        weight (Tensor): weight, shape (out_features, in_features).
        activation (str): "id" | "gelu" | "gelu_approx" | "squared_relu".
        act_input (Tensor|None): pre-activation input, shape (..., in_features);
            required when activation != "id".

    Returns:
        Tensor: grad_input, shape (..., in_features).
    """
    return _m.triton_dgrad_act(
        grad_output, weight, activation=activation, act_input=act_input)


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: triton_dgrad_act requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: triton_dgrad_act requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    for dtype in (torch.float16, torch.bfloat16):
        # activation="id": grad_input = grad_output @ weight
        T, OUT, IN = 128, 256, 512
        grad_output = torch.randn(T, OUT, device=device, dtype=dtype) * 0.1
        weight = torch.randn(OUT, IN, device=device, dtype=dtype) * 0.1

        gi = triton_dgrad_act(grad_output, weight, activation="id")
        ref = grad_output.float() @ weight.float()
        torch.testing.assert_close(gi.float(), ref, rtol=3e-2, atol=3e-2)
        print(f"pass: triton_dgrad_act id dtype={dtype}")

        # activation="squared_relu": multiply by d/dz (relu(z)^2) = 2*relu(z)
        act_input = torch.randn(T, IN, device=device, dtype=dtype) * 0.5
        gi2 = triton_dgrad_act(grad_output, weight, activation="squared_relu", act_input=act_input)
        dact = 2.0 * torch.nn.functional.relu(act_input.float())
        ref2 = (grad_output.float() @ weight.float()) * dact
        torch.testing.assert_close(gi2.float(), ref2, rtol=3e-2, atol=3e-2)
        print(f"pass: triton_dgrad_act squared_relu dtype={dtype}")

    print("\ntriton_dgrad_act all tests passed")
