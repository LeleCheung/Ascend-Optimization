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


def pairwise_l2(xq, xb):
    """faiss full pairwise squared-L2 distance matrix.

    CUDA operator (baseline). On GPU faiss computes this via:
        faiss/gpu/impl/Distance.cu (runL2Distance) -> l2NormRowMajor
        (faiss/gpu/impl/L2Norm.cu) for the row-major query/database norms,
        plus l2NormColMajor (faiss/gpu/impl/L2Norm.cu) for the transposed
        column-major layout used when the base vectors are stored column-major,
        + a GEMM for the -2 * xq . xb^T cross term. The squared norms of xq and
        xb are then broadcast-added onto the GEMM result to form
        ||xq||^2 - 2 xq.xb^T + ||xb||^2.
    On CPU it dispatches to pairwise_L2sqr (faiss/utils/distances.cpp).

    Args:
        xq (Tensor): query vectors, shape (nq, d), float32.
        xb (Tensor): database vectors, shape (nb, d), float32.

    Returns:
        Tensor: squared-L2 distance matrix, shape (nq, nb), float32, on the
        same device as the inputs.
    """
    if faiss is None:
        raise RuntimeError("pairwise_l2 baseline requires faiss to be installed")

    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2, f"dim mismatch: xq d={d}, xb d={d2}"

    xq_c = xq.contiguous().float()
    xb_c = xb.contiguous().float()

    import numpy as np

    xq_np = np.ascontiguousarray(xq_c.detach().cpu().numpy(), dtype="float32")
    xb_np = np.ascontiguousarray(xb_c.detach().cpu().numpy(), dtype="float32")
    # faiss.pairwise_distances returns squared-L2 distances, shape (nq, nb).
    Dm = faiss.pairwise_distances(xq_np, xb_np)
    return torch.from_numpy(np.ascontiguousarray(Dm)).to(xq.device)


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: pairwise_l2 requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    def ref_pairwise_l2(xq, xb):
        return torch.cdist(xq.float(), xb.float(), p=2) ** 2

    nq, nb, d = 128, 2048, 64
    xq = torch.randn(nq, d, device=device, dtype=dtype)
    xb = torch.randn(nb, d, device=device, dtype=dtype)

    D = pairwise_l2(xq, xb)
    D_ref = ref_pairwise_l2(xq, xb)

    assert D.shape == (nq, nb), D.shape
    torch.testing.assert_close(D.cpu(), D_ref.cpu(), rtol=1e-3, atol=1e-2)
    print("pairwise_l2 OK: distance matrix matches torch.cdist reference")
