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
    from flash_attn.ops import rms_norm as _m
except (ModuleNotFoundError, ImportError):
    _m = None


def dropout_add_rms_norm(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    """FlashAttention baseline for dropout_add_rms_norm
    (fused kernel from csrc/layer_norm).

    Fuses dropout + residual add + RMSNorm:
        x = dropout(x0) + residual
        z = RMSNorm(x, weight, eps)   # no mean subtraction

    Args:
        x0 (Tensor):    input to dropout, shape (..., hidden_size).
        residual (Tensor|None): optional residual to add, same shape as x0.
        weight (Tensor): RMSNorm gain, shape (hidden_size,).
        bias (Tensor|None): optional bias, shape (hidden_size,).
        dropout_p (float): dropout probability on x0 (0.0 to disable).
        epsilon (float): RMSNorm eps.
        rowscale (Tensor|None): optional per-row scale applied to x0.
        layerscale (Tensor|None): optional per-channel scale applied to x0.
        prenorm (bool): if True, also return the pre-norm sum x.
        residual_in_fp32 (bool): keep residual accumulation in fp32 (only
            takes effect when residual is None).
        return_dropout_mask (bool): if True, also return the dropout mask.

    Returns:
        z by default. With prenorm/return_dropout_mask enabled, the pre-norm
        tensor and/or dropout mask are appended (see kernel docs).
    """
    return _m.dropout_add_rms_norm(
        x0,
        residual,
        weight,
        bias,
        dropout_p,
        epsilon,
        rowscale=rowscale,
        layerscale=layerscale,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=return_dropout_mask,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: dropout_add_rms_norm requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: dropout_add_rms_norm requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"
    dtype = torch.float16
    epsilon = 1e-5
    M, N = 128, 512

    def ref_rms_norm(x, w, eps):
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return xf * rms * w.float()

    x0 = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=dtype)

    # ---- Case 1: with residual, dropout_p=0.0 (deterministic) ----
    z = dropout_add_rms_norm(x0, residual, w, None, 0.0, epsilon)
    ref = ref_rms_norm(x0.float() + residual.float(), w, epsilon)
    torch.testing.assert_close(z.float(), ref, rtol=2e-2, atol=2e-2)
    print("pass: dropout_add_rms_norm with residual")

    # ---- Case 2: no residual ----
    z2 = dropout_add_rms_norm(x0, None, w, None, 0.0, epsilon)
    ref2 = ref_rms_norm(x0, w, epsilon)
    torch.testing.assert_close(z2.float(), ref2, rtol=2e-2, atol=2e-2)
    print("pass: dropout_add_rms_norm without residual")

    # ---- Case 3: prenorm returns residual sum ----
    out = dropout_add_rms_norm(x0, residual, w, None, 0.0, epsilon, prenorm=True)
    z3, x_prenorm = out[0], out[1]
    torch.testing.assert_close(
        x_prenorm.float(), x0.float() + residual.float(), rtol=2e-2, atol=2e-2)
    print("pass: prenorm returns residual sum")

    print("\ndropout_add_rms_norm all tests passed")
