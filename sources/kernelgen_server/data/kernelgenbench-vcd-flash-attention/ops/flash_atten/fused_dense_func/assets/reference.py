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
    from flash_attn.ops import fused_dense as _m
except (ModuleNotFoundError, ImportError):
    _m = None


def fused_dense_func(
    x,
    weight,
    bias=None,
    return_residual=False,
    process_group=None,
    sequence_parallel=True,
):
    """FlashAttention baseline for fused_dense_func
    (csrc/fused_dense_lib, cuBLASLt).

    Fused linear layer: out = x @ weight.T + bias, with an optional residual
    return for fused add patterns.

    Args:
        x (Tensor): input, shape (..., in_features).
        weight (Tensor): weight, shape (out_features, in_features).
        bias (Tensor|None): bias, shape (out_features,).
        return_residual (bool): if True, also return x (for fused residual).
        process_group (ProcessGroup|None): tensor-parallel group.
        sequence_parallel (bool): sequence-parallel mode flag.

    Returns:
        Tensor out, or (out, x) when return_residual=True.
    """
    return _m.fused_dense_func(
        x, weight, bias,
        return_residual=return_residual,
        process_group=process_group,
        sequence_parallel=sequence_parallel,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: fused_dense_func requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: fused_dense_func requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    for dtype in (torch.float16, torch.bfloat16):
        for has_bias in (True, False):
            T, IN, OUT = 128, 512, 256
            x = torch.randn(T, IN, device=device, dtype=dtype)
            w = torch.randn(OUT, IN, device=device, dtype=dtype)
            b = torch.randn(OUT, device=device, dtype=dtype) if has_bias else None

            out = fused_dense_func(x, w, b)
            ref = torch.nn.functional.linear(x.float(), w.float(), b.float() if b is not None else None)
            torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
            print(f"pass: fused_dense_func dtype={dtype} bias={has_bias}")

    # return_residual returns (out, x)
    x = torch.randn(64, 512, device=device, dtype=torch.float16)
    w = torch.randn(256, 512, device=device, dtype=torch.float16)
    out, res = fused_dense_func(x, w, None, return_residual=True)
    torch.testing.assert_close(res, x)
    print("pass: fused_dense_func return_residual")

    print("\nfused_dense_func all tests passed")
