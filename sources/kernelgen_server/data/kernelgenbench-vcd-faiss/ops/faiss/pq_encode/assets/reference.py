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


def pq_encode(x, centroids):
    """Product-quantizer encoding.

    CUDA operator (baseline). The faiss C++/CUDA implementation lives in
        faiss/gpu/impl/PQCodeDistances.cu (code assignment) and
        faiss/impl/ProductQuantizer.cpp::compute_codes
    Each input vector is split into M contiguous sub-vectors; every sub-vector
    is assigned the index of its nearest centroid inside that sub-space.

    faiss has no GPU Python entry point for standalone PQ encoding, so the
    encoding always runs through faiss' CPU kernels. Inputs are moved to CPU
    for faiss and the result is returned on the original device.

    Args:
        x (Tensor): vectors to encode, shape (n, d), float32. d == M * dsub.
        centroids (Tensor): per-subspace codebooks, shape (M, ksub, dsub),
            float32. M sub-quantizers, ksub = 2**nbits centroids each.

    Returns:
        Tensor: PQ codes, shape (n, M), uint8 (requires ksub <= 256).
    """
    if faiss is None:
        raise RuntimeError("pq_encode baseline requires faiss to be installed")

    import numpy as np

    M, ksub, dsub = centroids.shape
    d = M * dsub
    n = x.shape[0]
    assert x.shape[1] == d, f"dim mismatch: x d={x.shape[1]}, M*dsub={d}"
    nbits = int(round(float(torch.log2(torch.tensor(float(ksub))))))
    assert 2 ** nbits == ksub, f"ksub={ksub} must be a power of two"
    assert ksub <= 256, "uint8 codes require ksub <= 256"

    x_np = np.ascontiguousarray(x.detach().cpu().numpy(), dtype="float32")
    cen_np = np.ascontiguousarray(centroids.detach().cpu().numpy(), dtype="float32")

    pq = faiss.ProductQuantizer(d, M, nbits)
    # Inject the supplied codebooks so the mapping is fully determined by the
    # inputs (no random training) -> a Triton solution can reproduce it exactly.
    faiss.copy_array_to_vector(cen_np.ravel(), pq.centroids)
    codes = pq.compute_codes(x_np)  # (n, M) uint8

    return torch.from_numpy(np.ascontiguousarray(codes)).to(x.device)


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: pq_encode requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def ref_pq_encode(x, centroids):
        # Independent reference: nearest centroid per sub-space (argmin L2).
        n, d = x.shape
        M, ksub, dsub = centroids.shape
        xsub = x.reshape(n, M, dsub)              # (n, M, dsub)
        # (n, M, ksub) squared distances to each centroid
        dist = ((xsub.unsqueeze(2) - centroids.unsqueeze(0)) ** 2).sum(dim=-1)
        return dist.argmin(dim=-1).to(torch.uint8)

    n, M, ksub, dsub = 100, 8, 256, 8
    d = M * dsub
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    centroids = torch.randn(M, ksub, dsub, device=device, dtype=torch.float32)

    codes = pq_encode(x, centroids)
    ref = ref_pq_encode(x, centroids)

    assert codes.shape == (n, M) and codes.dtype == torch.uint8, (codes.shape, codes.dtype)
    agreement = (codes.cpu() == ref.cpu()).float().mean().item()
    assert agreement == 1.0, f"pq_encode disagrees with torch reference: {agreement:.4f}"
    print(f"pq_encode OK: encoded {n} vectors to {tuple(codes.shape)}, code agreement = {agreement:.4f}")
