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
    from flash_attn.ops import layer_norm as _m
except (ModuleNotFoundError, ImportError):
    _m = None


def dropout_add_layer_norm_parallel_residual(
    x0,
    x1,
    residual,
    weight0,
    bias0,
    weight1,
    bias1,
    dropout_p,
    epsilon,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    """FlashAttention baseline for dropout_add_layer_norm_parallel_residual
    (fused kernel from csrc/layer_norm).

    Computes two parallel LayerNorm branches over a shared residual sum:
        x  = dropout(x0) + [dropout(x1)] + [residual]
        z0 = LayerNorm(x, weight0, bias0, eps)
        z1 = LayerNorm(x, weight1, bias1, eps)   # only if weight1 is not None

    Args:
        x0 (Tensor):    first input, shape (..., hidden_size).
        x1 (Tensor|None): optional second input to add, same shape as x0.
        residual (Tensor|None): optional residual to add, same shape as x0.
        weight0 (Tensor): gain for branch 0, shape (hidden_size,).
        bias0 (Tensor|None): bias for branch 0, shape (hidden_size,).
        weight1 (Tensor|None): gain for branch 1; if None, z1 is None.
        bias1 (Tensor|None): bias for branch 1, shape (hidden_size,).
        dropout_p (float): dropout probability on x0/x1 (0.0 to disable).
        epsilon (float): LayerNorm eps.
        prenorm (bool): if True, also return the pre-norm residual sum x.
        residual_in_fp32 (bool): keep residual accumulation in fp32 (only
            takes effect when residual is None).
        return_dropout_mask (bool): if True, also return the dropout masks.

    Returns:
        (z0, z1) by default. With prenorm/return_dropout_mask enabled, the
        pre-norm tensor and/or dropout masks are appended (see kernel docs).
    """
    return _m.dropout_add_layer_norm_parallel_residual(
        x0,
        x1,
        residual,
        weight0,
        bias0,
        weight1,
        bias1,
        dropout_p,
        epsilon,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=return_dropout_mask,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: dropout_add_layer_norm_parallel_residual requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: dropout_add_layer_norm_parallel_residual requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"
    dtype = torch.float16
    epsilon = 1e-5

    def ref_parallel_residual(x0, x1, residual, w0, b0, w1, b1, eps):
        # Reference with dropout_p = 0.0 (deterministic): identity dropout.
        x = x0.float()
        if x1 is not None:
            x = x + x1.float()
        if residual is not None:
            x = x + residual.float()
        z0 = torch.nn.functional.layer_norm(
            x, (x.shape[-1],), w0.float(), b0.float() if b0 is not None else None, eps)
        z1 = None
        if w1 is not None:
            z1 = torch.nn.functional.layer_norm(
                x, (x.shape[-1],), w1.float(), b1.float() if b1 is not None else None, eps)
        return z0, z1

    # ---- Case 1: full inputs, two parallel branches ----
    M, N = 128, 512
    x0 = torch.randn(M, N, device=device, dtype=dtype)
    x1 = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    w0 = torch.randn(N, device=device, dtype=dtype)
    b0 = torch.randn(N, device=device, dtype=dtype)
    w1 = torch.randn(N, device=device, dtype=dtype)
    b1 = torch.randn(N, device=device, dtype=dtype)

    z0, z1 = dropout_add_layer_norm_parallel_residual(
        x0, x1, residual, w0, b0, w1, b1, 0.0, epsilon)
    ref_z0, ref_z1 = ref_parallel_residual(x0, x1, residual, w0, b0, w1, b1, epsilon)
    torch.testing.assert_close(z0.float(), ref_z0, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(z1.float(), ref_z1, rtol=2e-2, atol=2e-2)
    print("pass: two-branch parallel residual")

    # ---- Case 2: no x1, no residual (single branch add only) ----
    z0b, z1b = dropout_add_layer_norm_parallel_residual(
        x0, None, None, w0, b0, None, None, 0.0, epsilon)
    ref_z0b, _ = ref_parallel_residual(x0, None, None, w0, b0, None, None, epsilon)
    torch.testing.assert_close(z0b.float(), ref_z0b, rtol=2e-2, atol=2e-2)
    assert z1b is None, "z1 should be None when weight1 is None"
    print("pass: single-branch, no x1/residual")

    # ---- Case 3: prenorm returns the pre-norm residual sum ----
    out = dropout_add_layer_norm_parallel_residual(
        x0, x1, residual, w0, b0, w1, b1, 0.0, epsilon, prenorm=True)
    z0c, z1c, x_prenorm = out[0], out[1], out[2]
    ref_x = (x0.float() + x1.float() + residual.float())
    torch.testing.assert_close(x_prenorm.float(), ref_x, rtol=2e-2, atol=2e-2)
    print("pass: prenorm returns residual sum")

    print("\ndropout_add_layer_norm_parallel_residual all tests passed")
