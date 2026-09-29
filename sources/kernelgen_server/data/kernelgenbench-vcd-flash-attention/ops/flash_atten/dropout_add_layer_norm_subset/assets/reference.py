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


def dropout_add_layer_norm_subset(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    layerscale=None,
    x0_subset=None,
    out_subset=None,
    rowscale_const=1.0,
    out_numrows=0,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    """FlashAttention baseline for dropout_add_layer_norm_subset
    (fused kernel from csrc/layer_norm).

    Like dropout_add_layer_norm, but computes only on a subset of rows
    (gathered via x0_subset / scattered via out_subset). Passing no subset
    indices makes it behave like the dense dropout_add_layer_norm:
        x = rowscale_const * dropout(x0) + residual
        z = LayerNorm(x, weight, bias, eps)

    Args:
        x0 (Tensor):    input to dropout, shape (num_rows, hidden_size).
        residual (Tensor|None): optional residual to add.
        weight (Tensor): LayerNorm gain, shape (hidden_size,).
        bias (Tensor|None): LayerNorm bias, shape (hidden_size,).
        dropout_p (float): dropout probability on x0 (0.0 to disable).
        epsilon (float): LayerNorm eps.
        layerscale (Tensor|None): optional per-channel scale applied to x0.
        x0_subset (Tensor|None): row indices to gather from x0 (1-based; 0 skips).
        out_subset (Tensor|None): row indices to scatter results into output.
        rowscale_const (float): constant scale applied to x0.
        out_numrows (int): number of output rows when out_subset is used.
        prenorm (bool): if True, also return the pre-norm sum x.
        residual_in_fp32 (bool): keep residual accumulation in fp32.
        return_dropout_mask (bool): if True, also return the dropout mask.

    Returns:
        z by default. With prenorm/return_dropout_mask enabled, the pre-norm
        tensor and/or dropout mask are appended (see kernel docs).
    """
    return _m.dropout_add_layer_norm_subset(
        x0,
        residual,
        weight,
        bias,
        dropout_p,
        epsilon,
        layerscale=layerscale,
        x0_subset=x0_subset,
        out_subset=out_subset,
        rowscale_const=rowscale_const,
        out_numrows=out_numrows,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=return_dropout_mask,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: dropout_add_layer_norm_subset requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: dropout_add_layer_norm_subset requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"
    dtype = torch.float16
    epsilon = 1e-5
    M, N = 128, 512

    x0 = torch.randn(M, N, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)

    # ---- Case 1: dense (no subset indices), dropout_p=0.0 (deterministic) ----
    z = dropout_add_layer_norm_subset(x0, residual, w, b, 0.0, epsilon)
    ref = torch.nn.functional.layer_norm(
        (x0.float() + residual.float()), (N,), w.float(), b.float(), epsilon)
    torch.testing.assert_close(z.float(), ref, rtol=2e-2, atol=2e-2)
    print("pass: dense dropout_add_layer_norm_subset")

    # ---- Case 2: gather/scatter via x0_subset / out_subset (dropout_p=0.0) ----
    # subset indices are 1-based; 0 means "skip this row". x0_subset gathers the
    # compacted x0 rows, out_subset scatters results into the compacted output;
    # residual keeps the full (num_full_rows) layout.
    num_full = 8
    x0_mask = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0], device=device, dtype=torch.bool)
    out_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 1], device=device, dtype=torch.bool)
    out_numrows = int(out_mask.sum().item())
    x0_subset = torch.cumsum(x0_mask.int(), 0).masked_fill_(~x0_mask, 0).to(torch.int32)
    out_subset = torch.cumsum(out_mask.int(), 0).masked_fill_(~out_mask, 0).to(torch.int32)

    x0_full = torch.randn(num_full, N, device=device, dtype=dtype)
    res_full = torch.randn(num_full, N, device=device, dtype=dtype)
    x0_c = x0_full[x0_mask].contiguous()  # compacted input rows

    z2 = dropout_add_layer_norm_subset(
        x0_c, res_full, w, b, 0.0, epsilon,
        x0_subset=x0_subset, out_subset=out_subset,
        rowscale_const=1.0, out_numrows=out_numrows,
    )
    # Reference: for each full row i with out_subset[i] > 0, the compacted output
    # row (out_subset[i]-1) is LayerNorm(x0_row + res_full[i]), where x0_row is the
    # gathered x0 row (x0_subset[i]-1) or zero when x0_subset[i] == 0.
    ref2 = torch.zeros(out_numrows, N, device=device, dtype=torch.float32)
    for i in range(num_full):
        o = int(out_subset[i].item())
        if o == 0:
            continue
        xi = int(x0_subset[i].item())
        x_row = x0_c[xi - 1].float() if xi > 0 else torch.zeros(N, device=device)
        ref2[o - 1] = torch.nn.functional.layer_norm(
            (x_row + res_full[i].float()), (N,), w.float(), b.float(), epsilon)
    torch.testing.assert_close(z2.float(), ref2, rtol=2e-2, atol=2e-2)
    print("pass: gather/scatter subset")

    print("\ndropout_add_layer_norm_subset all tests passed")
