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
    from flash_attn.ops.triton import layer_norm as _m
except (ModuleNotFoundError, ImportError):
    _m = None


def triton_layer_norm_linear_fn(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias=None,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    """FlashAttention Triton baseline for layer_norm_linear_fn.

    Fuses (residual add +) LayerNorm/RMSNorm followed by a linear projection:
        x_res = x + [residual]
        h = LayerNorm(x_res, norm_weight, norm_bias, eps)
        out = h @ linear_weight.T + linear_bias

    Args:
        x (Tensor): input, shape (..., hidden_size).
        norm_weight (Tensor): norm gain, shape (hidden_size,).
        norm_bias (Tensor|None): norm bias, shape (hidden_size,).
        linear_weight (Tensor): linear weight, shape (out_features, hidden_size).
        linear_bias (Tensor|None): linear bias, shape (out_features,).
        residual (Tensor|None): optional residual to add before the norm.
        eps (float): norm eps.
        prenorm (bool): if True, also return the pre-norm residual sum.
        residual_in_fp32 (bool): accumulate residual in fp32.
        is_rms_norm (bool): use RMSNorm instead of LayerNorm.

    Returns:
        Tensor out by default; the pre-norm tensor is appended when prenorm=True.
    """
    return _m.layer_norm_linear_fn(
        x, norm_weight, norm_bias, linear_weight, linear_bias,
        residual=residual, eps=eps, prenorm=prenorm,
        residual_in_fp32=residual_in_fp32, is_rms_norm=is_rms_norm,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: triton_layer_norm_linear_fn requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: triton_layer_norm_linear_fn requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"
    eps = 1e-5

    for dtype in (torch.float16, torch.bfloat16):
        T, H, OUT = 128, 512, 256
        x = torch.randn(T, H, device=device, dtype=dtype)
        nw = torch.randn(H, device=device, dtype=dtype)
        nb = torch.randn(H, device=device, dtype=dtype)
        lw = torch.randn(OUT, H, device=device, dtype=dtype) * 0.1
        lb = torch.randn(OUT, device=device, dtype=dtype) * 0.1

        out = triton_layer_norm_linear_fn(x, nw, nb, lw, lb, eps=eps)
        h = torch.nn.functional.layer_norm(x.float(), (H,), nw.float(), nb.float(), eps)
        ref = torch.nn.functional.linear(h, lw.float(), lb.float())
        torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)
        print(f"pass: triton_layer_norm_linear_fn dtype={dtype}")

    print("\ntriton_layer_norm_linear_fn all tests passed")
