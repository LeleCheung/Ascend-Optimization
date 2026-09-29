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


def triton_linear_act(x, weight, bias=None, activation="id", save_act_input=False):
    """FlashAttention Triton baseline for triton_linear_act.

    Fused linear + activation forward:
        out = activation(x @ weight.T + bias)

    Args:
        x (Tensor): input, shape (..., in_features).
        weight (Tensor): weight, shape (out_features, in_features).
        bias (Tensor|None): bias, shape (out_features,).
        activation (str): "id" | "gelu" | "gelu_approx" | "squared_relu" | "relu".
        save_act_input (bool): also return the pre-activation input (for backward).

    Returns:
        Tensor: result, shape (..., out_features)
        (or (out, act_input) when save_act_input=True).
    """
    return _m.triton_linear_act(
        x, weight, bias=bias, activation=activation, save_act_input=save_act_input)


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: triton_linear_act requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: triton_linear_act requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    def act_fn(h, name):
        if name == "id":
            return h
        if name == "gelu":
            return torch.nn.functional.gelu(h)
        if name == "gelu_approx":
            return torch.nn.functional.gelu(h, approximate="tanh")
        if name == "squared_relu":
            return torch.nn.functional.relu(h) ** 2
        raise ValueError(name)

    for dtype in (torch.float16, torch.bfloat16):
        for activation in ("id", "gelu", "gelu_approx", "squared_relu"):
            T, IN, OUT = 128, 512, 256
            x = torch.randn(T, IN, device=device, dtype=dtype) * 0.5
            w = torch.randn(OUT, IN, device=device, dtype=dtype) * 0.1
            b = torch.randn(OUT, device=device, dtype=dtype) * 0.1

            out = triton_linear_act(x, w, b, activation=activation)
            h = torch.nn.functional.linear(x.float(), w.float(), b.float())
            ref = act_fn(h, activation)
            torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)
            print(f"pass: triton_linear_act dtype={dtype} activation={activation}")

    print("\ntriton_linear_act all tests passed")
