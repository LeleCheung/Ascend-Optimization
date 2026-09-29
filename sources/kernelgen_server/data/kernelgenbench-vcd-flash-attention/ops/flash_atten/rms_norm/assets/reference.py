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


def rms_norm(x, weight, epsilon):
    """FlashAttention baseline for rms_norm (fused kernel from csrc/layer_norm).

    Computes RMSNorm over the last dimension (no mean subtraction, no bias):
        z = x / sqrt(mean(x^2) + eps) * weight

    Args:
        x (Tensor):      input, shape (..., hidden_size).
        weight (Tensor): gain, shape (hidden_size,).
        epsilon (float): numerical eps.

    Returns:
        Tensor: normalized output, same shape as x.
    """
    return _m.rms_norm(x, weight, epsilon)


if __name__ == "__main__":
    torch.manual_seed(0)

    if _m is None:
        print("skip: rms_norm requires flash_attn to be installed")
        raise SystemExit(0)
    if not torch.cuda.is_available():
        print("skip: rms_norm requires a CUDA device")
        raise SystemExit(0)

    device = "cuda"
    epsilon = 1e-5

    def ref_rms_norm(x, w, eps):
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return xf * rms * w.float()

    for dtype in (torch.float16, torch.bfloat16):
        M, N = 128, 512
        x = torch.randn(M, N, device=device, dtype=dtype)
        w = torch.randn(N, device=device, dtype=dtype)

        z = rms_norm(x, w, epsilon)
        ref = ref_rms_norm(x, w, epsilon)
        torch.testing.assert_close(z.float(), ref, rtol=2e-2, atol=2e-2)
        print(f"pass: rms_norm {dtype}")

    print("\nrms_norm all tests passed")
