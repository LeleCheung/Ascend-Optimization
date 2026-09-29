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
    import faiss
except (ModuleNotFoundError, ImportError):
    faiss = None


def normalize_l2(x):
    """faiss row-wise L2 normalization: each row is scaled to unit L2 norm.

    CUDA operator (baseline):
        faiss/gpu/impl/L2Norm.cu -> l2NormRowMajor
    faiss computes the per-row L2 norm and divides each row by it in place.

    Args:
        x (Tensor): input vectors, shape (n, d), float32.

    Returns:
        Tensor: L2-normalized vectors, shape (n, d), float32, on the same
        device as ``x``.
    """
    if faiss is None:
        raise RuntimeError("normalize_l2 baseline requires faiss to be installed")

    import numpy as np

    x_np = np.ascontiguousarray(x.detach().cpu().numpy(), dtype="float32")
    faiss.normalize_L2(x_np)  # in-place row-wise L2 normalization
    return torch.from_numpy(x_np).to(x.device)


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: normalize_l2 requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, d = 128, 64
    x = torch.randn(n, d, device=device, dtype=torch.float32)

    x_norm = normalize_l2(x)
    # Independent torch reference: divide each row by its L2 norm.
    x_ref = x / x.norm(dim=1, keepdim=True)

    assert x_norm.shape == (n, d), x_norm.shape
    torch.testing.assert_close(x_norm.cpu(), x_ref.cpu(), rtol=1e-4, atol=1e-5)
    row_norms = x_norm.norm(dim=1)
    torch.testing.assert_close(row_norms.cpu(), torch.ones(n), rtol=1e-4, atol=1e-5)
    print("normalize_l2 OK: rows unit-norm, matches torch reference")
