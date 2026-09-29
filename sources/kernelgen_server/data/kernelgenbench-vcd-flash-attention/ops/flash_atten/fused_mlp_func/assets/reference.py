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


def fused_mlp_func(
    x,
    weight1,
    weight2,
    bias1=None,
    bias2=None,
    activation="gelu_approx",
    save_pre_act=True,
    return_residual=False,
    checkpoint_lvl=0,
    heuristic=0,
    process_group=None,
    sequence_parallel=True,
):
    """FlashAttention baseline for fused_mlp_func
    (csrc/fused_dense_lib, cuBLASLt).

    Two-layer MLP with a fused activation:
        h = activation(x @ weight1.T + bias1)
        out = h @ weight2.T + bias2

    Args:
        x (Tensor): input, shape (..., in_features).
        weight1 (Tensor): first weight, shape (hidden, in_features).
        weight2 (Tensor): second weight, shape (out_features, hidden).
        bias1 (Tensor|None): first bias, shape (hidden,).
        bias2 (Tensor|None): second bias, shape (out_features,).
        activation (str): "gelu_approx" | "relu" | "sqrelu".
        save_pre_act (bool): save pre-activation for backward.
        return_residual (bool): if True, also return x.
        checkpoint_lvl (int): activation-recompute level for backward.
        heuristic (int): cuBLASLt heuristic selector (-1 auto).
        process_group (ProcessGroup|None): tensor-parallel group.
        sequence_parallel (bool): sequence-parallel mode flag.

    Returns:
        Tensor out, or (out, x) when return_residual=True.
    """
    return _m.fused_mlp_func(
        x, weight1, weight2, bias1, bias2,
        activation=activation, save_pre_act=save_pre_act,
        return_residual=return_residual, checkpoint_lvl=checkpoint_lvl,
        heuristic=heuristic, process_group=process_group,
        sequence_parallel=sequence_parallel,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: fused_mlp_func requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: fused_mlp_func requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"

    def act_fn(h, name):
        if name == "gelu_approx":
            return torch.nn.functional.gelu(h, approximate="tanh")
        if name == "relu":
            return torch.nn.functional.relu(h)
        if name == "sqrelu":
            return torch.nn.functional.relu(h) ** 2
        raise ValueError(name)

    for dtype in (torch.float16, torch.bfloat16):
        for activation in ("gelu_approx", "relu", "sqrelu"):
            # in/hidden divisible by 128 to satisfy save_pre_act constraints
            T, IN, HID = 128, 512, 1024
            x = torch.randn(T, IN, device=device, dtype=dtype)
            w1 = torch.randn(HID, IN, device=device, dtype=dtype) * 0.1
            b1 = torch.randn(HID, device=device, dtype=dtype) * 0.1
            w2 = torch.randn(IN, HID, device=device, dtype=dtype) * 0.1
            b2 = torch.randn(IN, device=device, dtype=dtype) * 0.1

            # sqrelu is only implemented on the heuristic == -1 code path.
            heuristic = -1 if activation == "sqrelu" else 0
            out = fused_mlp_func(x, w1, w2, b1, b2, activation=activation, heuristic=heuristic)
            h = act_fn(torch.nn.functional.linear(x.float(), w1.float(), b1.float()), activation)
            ref = torch.nn.functional.linear(h, w2.float(), b2.float())
            # Two chained matmuls in low precision drift from the fp32 reference.
            # bf16 (8-bit mantissa) drifts far more than fp16, and sqrelu squares
            # the activation, amplifying the error further, so widen accordingly.
            if dtype == torch.float16:
                rtol = atol = 5e-2
            else:  # bfloat16
                rtol, atol = (2e-1, 6e-1) if activation == "sqrelu" else (1e-1, 2e-1)
            torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)
            print(f"pass: fused_mlp_func dtype={dtype} activation={activation}")

    print("\nfused_mlp_func all tests passed")
