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


def pq_decode(codes, centroids):
    """Product-quantizer decoding / vector reconstruction from PQ codes.

    CUDA operators (baseline). This one baseline covers faiss' reconstruction
    kernels, all of which gather sub-space centroids by code and concatenate:
        faiss/gpu/impl/IVFUtils.cu / faiss/gpu/impl/VectorResidual.cu
            gatherReconstructByIds    -- reconstruct a scattered set of ids
            gatherReconstructByRange  -- reconstruct a contiguous [start,end)
        faiss/impl/ProductQuantizer.cpp::decode
    "Decode a set of codes" is exactly "reconstruct by ids", and decoding a
    contiguous slice of the code table is "reconstruct by range"; both are the
    same gather-and-concat, so pq_decode is the single semantic operator for
    them. faiss exposes only a CPU decode entry point, so decoding runs on CPU
    and the result is returned on the input device.

    Args:
        codes (Tensor): PQ codes, shape (n, M), uint8.
        centroids (Tensor): per-subspace codebooks, shape (M, ksub, dsub),
            float32.

    Returns:
        Tensor: reconstructed vectors, shape (n, d) with d = M * dsub, float32.
    """
    if faiss is None:
        raise RuntimeError("pq_decode baseline requires faiss to be installed")

    import numpy as np

    M, ksub, dsub = centroids.shape
    d = M * dsub
    assert codes.shape[1] == M, f"code width {codes.shape[1]} != M {M}"
    nbits = int(round(float(torch.log2(torch.tensor(float(ksub))))))
    assert 2 ** nbits == ksub, f"ksub={ksub} must be a power of two"

    codes_np = np.ascontiguousarray(codes.detach().cpu().numpy(), dtype=np.uint8)
    cen_np = np.ascontiguousarray(centroids.detach().cpu().numpy(), dtype="float32")

    pq = faiss.ProductQuantizer(d, M, nbits)
    faiss.copy_array_to_vector(cen_np.ravel(), pq.centroids)
    x_rec = pq.decode(codes_np)  # (n, d) float32

    return torch.from_numpy(np.ascontiguousarray(x_rec)).to(codes.device)


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: pq_decode requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def ref_pq_decode(codes, centroids):
        # Independent reference: gather each sub-space centroid by its code and
        # concatenate -> exactly what gatherReconstructBy{Ids,Range} do.
        n, M = codes.shape
        Mc, ksub, dsub = centroids.shape
        idx = codes.long()                                   # (n, M)
        marange = torch.arange(M, device=codes.device)
        gathered = centroids[marange.unsqueeze(0), idx]      # (n, M, dsub)
        return gathered.reshape(n, M * dsub).to(torch.float32)

    n, M, ksub, dsub = 100, 8, 256, 8
    d = M * dsub
    codes = torch.randint(0, ksub, (n, M), device=device, dtype=torch.uint8)
    centroids = torch.randn(M, ksub, dsub, device=device, dtype=torch.float32)

    rec = pq_decode(codes, centroids)
    ref = ref_pq_decode(codes, centroids)

    assert rec.shape == (n, d) and rec.dtype == torch.float32, (rec.shape, rec.dtype)
    torch.testing.assert_close(rec.cpu(), ref.cpu(), rtol=1e-5, atol=1e-5)

    # Exercise the two reconstruction modes the CUDA kernels distinguish.
    ids = torch.tensor([3, 7, 15, 42], device=device)
    rec_ids = pq_decode(codes[ids], centroids)                # gatherReconstructByIds
    torch.testing.assert_close(rec_ids.cpu(), rec[ids.cpu()].cpu(), rtol=1e-5, atol=1e-5)
    rec_range = pq_decode(codes[10:20], centroids)            # gatherReconstructByRange
    torch.testing.assert_close(rec_range.cpu(), rec[10:20].cpu(), rtol=1e-5, atol=1e-5)

    print(
        f"pq_decode OK: reconstructed {tuple(rec.shape)}; "
        f"by-ids {tuple(rec_ids.shape)} and by-range {tuple(rec_range.shape)} match full decode"
    )
