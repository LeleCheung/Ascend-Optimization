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


def triton_layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    zero_centered_weight=False,
    is_rms_norm=False,
    return_dropout_mask=False,
    out_dtype=None,
    out=None,
    residual_out=None,
):
    """FlashAttention Triton baseline for layer_norm_fn.

    Fused (dropout + residual add +) LayerNorm implemented in Triton:
        x_res = dropout(x) + [x1] + [residual]
        z = LayerNorm(x_res, weight, bias, eps)

    Args:
        x (Tensor): input, shape (..., hidden_size).
        weight (Tensor): LayerNorm gain, shape (hidden_size,).
        bias (Tensor|None): LayerNorm bias, shape (hidden_size,).
        residual (Tensor|None): optional residual to add.
        x1 (Tensor|None): optional second input to add.
        weight1/bias1 (Tensor|None): optional second-branch norm params.
        eps (float): LayerNorm eps.
        dropout_p (float): dropout on x (0.0 to disable).
        rowscale (Tensor|None): optional per-row scale on x.
        prenorm (bool): if True, also return the pre-norm residual sum.
        residual_in_fp32 (bool): accumulate residual in fp32.
        zero_centered_weight (bool): treat weight as (1 + weight).
        is_rms_norm (bool): use RMSNorm instead of LayerNorm.
        return_dropout_mask (bool): also return the dropout mask.
        out_dtype (torch.dtype|None): output dtype.
        out/residual_out (Tensor|None): optional pre-allocated buffers.

    Returns:
        Tensor z by default; extra tensors appended for prenorm/dropout mask.
    """
    kwargs = dict(
        residual=residual, x1=x1, weight1=weight1, bias1=bias1,
        eps=eps, dropout_p=dropout_p, rowscale=rowscale, prenorm=prenorm,
        residual_in_fp32=residual_in_fp32, is_rms_norm=is_rms_norm,
        return_dropout_mask=return_dropout_mask, out=out, residual_out=residual_out,
    )
    # Optional kwargs only present in newer flash_attn versions; pass through
    # when the installed layer_norm_fn actually accepts them.
    import inspect
    accepted = inspect.signature(_m.layer_norm_fn).parameters
    if "zero_centered_weight" in accepted:
        kwargs["zero_centered_weight"] = zero_centered_weight
    if "out_dtype" in accepted:
        kwargs["out_dtype"] = out_dtype
    return _m.layer_norm_fn(x, weight, bias, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: triton_layer_norm_fn requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: triton_layer_norm_fn requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"
    eps = 1e-5

    for dtype in (torch.float16, torch.bfloat16):
        M, N = 128, 512
        x = torch.randn(M, N, device=device, dtype=dtype)
        w = torch.randn(N, device=device, dtype=dtype)
        b = torch.randn(N, device=device, dtype=dtype)

        # ---- plain LayerNorm (no residual, dropout_p=0.0) ----
        z = triton_layer_norm_fn(x, w, b, eps=eps)
        ref = torch.nn.functional.layer_norm(x.float(), (N,), w.float(), b.float(), eps)
        torch.testing.assert_close(z.float(), ref, rtol=2e-2, atol=2e-2)
        print(f"pass: triton_layer_norm_fn dtype={dtype}")

        # ---- with residual add ----
        residual = torch.randn(M, N, device=device, dtype=dtype)
        z2 = triton_layer_norm_fn(x, w, b, residual=residual, eps=eps)
        ref2 = torch.nn.functional.layer_norm(
            (x.float() + residual.float()), (N,), w.float(), b.float(), eps)
        torch.testing.assert_close(z2.float(), ref2, rtol=2e-2, atol=2e-2)
        print(f"pass: triton_layer_norm_fn with residual dtype={dtype}")

    print("\ntriton_layer_norm_fn all tests passed")
