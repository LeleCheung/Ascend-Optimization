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


def pairwise_ip(xq, xb):
    """faiss full pairwise inner-product similarity matrix.

    CUDA operator (baseline). Computes xq @ xb.T via BLAS GEMM.
    Args:
        xq (Tensor): query vectors, shape (nq, d), float32.
        xb (Tensor): database vectors, shape (nb, d), float32.
    Returns:
        Tensor: inner-product similarity matrix, shape (nq, nb), float32.
    """
    if faiss is None:
        raise RuntimeError("pairwise_ip baseline requires faiss to be installed")

    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2, f"dim mismatch: xq d={d}, xb d={d2}"

    xq_c = xq.contiguous().float()
    xb_c = xb.contiguous().float()

    import numpy as np
    xq_np = np.ascontiguousarray(xq_c.detach().cpu().numpy(), dtype="float32")
    xb_np = np.ascontiguousarray(xb_c.detach().cpu().numpy(), dtype="float32")
    Dm = faiss.pairwise_distances(xq_np, xb_np, metric=faiss.METRIC_INNER_PRODUCT)
    return torch.from_numpy(np.ascontiguousarray(Dm)).to(xq.device)


if __name__ == "__main__":
    torch.manual_seed(0)
    if faiss is None:
        print("skip: pairwise_ip requires faiss"); raise SystemExit(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nq, nb, d = 128, 2048, 64
    xq = torch.randn(nq, d, device=device, dtype=torch.float32)
    xb = torch.randn(nb, d, device=device, dtype=torch.float32)
    D = pairwise_ip(xq, xb)
    D_ref = xq.float() @ xb.float().t()
    assert D.shape == (nq, nb), D.shape
    torch.testing.assert_close(D.cpu(), D_ref.cpu(), rtol=1e-3, atol=1e-3)
    print("pairwise_ip OK: inner-product matrix matches reference")
