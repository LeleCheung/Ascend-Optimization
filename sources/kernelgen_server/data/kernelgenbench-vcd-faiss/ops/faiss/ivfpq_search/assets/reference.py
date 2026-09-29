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
    if hasattr(faiss, "StandardGpuResources"):
        import faiss.contrib.torch_utils  # noqa: F401
except (ModuleNotFoundError, ImportError):
    faiss = None

_GPU_RES = None


def _get_gpu_res():
    global _GPU_RES
    if _GPU_RES is None:
        _GPU_RES = faiss.StandardGpuResources()
    return _GPU_RES


def ivfpq_search(xq, xb, k, nlist=100, M=8, nbits=8, nprobe=10):
    """faiss IndexIVFPQ approximate nearest-neighbor search (L2 metric).

    IVFPQ combines an inverted file (coarse quantizer over ``nlist`` cells) with
    product quantization: each database vector's residual to its coarse centroid
    is split into ``M`` sub-vectors, each encoded to one of ``2**nbits`` codes.
    At query time a per-subquantizer distance lookup table (LUT) is built and the
    scan accumulates codeword distances (asymmetric distance computation), then
    selects the top-k. This subsumes the standalone interleaved-scan LUT lookup.

    CUDA operators (baseline). On GPU the scan runs in faiss' CUDA kernels:
        faiss/gpu/impl/PQScanMultiPassPrecomputed.cu::pqScanPrecomputedMultiPass
        faiss/gpu/impl/IVFInterleaved.cuh::pqScanPrecomputedInterleaved
        faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cu::pqScanNoPrecomputedMultiPass
        faiss/gpu/impl/IVFInterleaved.cuh::pqScanInterleaved  (interleaved LUT scan)
        faiss/gpu/impl/PQCodeDistances.cu::pqDistanceIPCorrection (IP metric fixup)

    Args:
        xq (Tensor):  query vectors, shape (nq, d), float32.
        xb (Tensor):  database vectors, shape (nb, d), float32.
        k (int):      number of nearest neighbors to return.
        nlist (int):  number of IVF cells (coarse centroids).
        M (int):      number of PQ sub-quantizers (must divide d).
        nbits (int):  bits per PQ code (codebook size = 2**nbits).
        nprobe (int): number of cells visited per query.

    Returns:
        (D, I) where
            D (Tensor): approximate squared-L2 distances, shape (nq, k), float32,
                        ascending along dim 1.
            I (Tensor): neighbor indices into xb, shape (nq, k), int64. Padding
                        value -1 may appear when a query finds fewer than k
                        candidates in the probed cells.

    Both outputs are torch tensors on the same device as the inputs.
    """
    if faiss is None:
        raise RuntimeError("ivfpq_search baseline requires faiss to be installed")

    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2, f"dim mismatch: xq d={d}, xb d={d2}"
    assert d % M == 0, f"M={M} must divide d={d}"
    nlist = min(nlist, nb)
    nprobe = min(nprobe, nlist)

    xq = xq.contiguous().float()
    xb = xb.contiguous().float()

    use_gpu = (
        xq.is_cuda
        and hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "GpuIndexIVFPQ")
    )

    if use_gpu:
        # GPU path -> exercises pqScan* / interleaved LUT scan kernels.
        res = _get_gpu_res()
        cfg = faiss.GpuIndexIVFPQConfig()
        index = faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits, faiss.METRIC_L2, cfg)
        index.train(xb)
        index.add(xb)
        index.nprobe = nprobe
        D, I = index.search(xq, k)
        return D.contiguous(), I.to(torch.int64).contiguous()

    # CPU fallback (numpy round-trip) so the reference still works without a GPU.
    import numpy as np

    xq_np = np.ascontiguousarray(xq.detach().cpu().numpy(), dtype="float32")
    xb_np = np.ascontiguousarray(xb.detach().cpu().numpy(), dtype="float32")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits, faiss.METRIC_L2)
    index.train(xb_np)
    index.add(xb_np)
    index.nprobe = nprobe
    D_np, I_np = index.search(xq_np, k)
    D = torch.from_numpy(np.ascontiguousarray(D_np)).to(xq.device)
    I = torch.from_numpy(np.ascontiguousarray(I_np)).to(xq.device).to(torch.int64)
    return D, I


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: ivfpq_search requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    def ref_bruteforce_knn(xq, xb, k):
        # Exact reference. IVFPQ is doubly approximate (IVF pruning + PQ
        # quantization), so we score it by recall, not equality.
        dist = torch.cdist(xq.float(), xb.float(), p=2) ** 2
        _, I = torch.topk(dist, k, dim=1, largest=False, sorted=True)
        return I.to(torch.int64)

    nq, nb, d, k = 64, 8192, 64, 10
    nlist, M, nbits, nprobe = 64, 8, 8, 24
    xq = torch.randn(nq, d, device=device, dtype=dtype)
    xb = torch.randn(nb, d, device=device, dtype=dtype)

    D, I = ivfpq_search(xq, xb, k, nlist=nlist, M=M, nbits=nbits, nprobe=nprobe)
    I_exact = ref_bruteforce_knn(xq, xb, k)

    assert D.shape == (nq, k) and I.shape == (nq, k), (D.shape, I.shape)
    hits = 0
    for q in range(nq):
        approx = set(I[q].cpu().tolist()) - {-1}
        exact = set(I_exact[q].cpu().tolist())
        hits += len(approx & exact)
    recall = hits / (nq * k)
    top1 = (I[:, 0].cpu() == I_exact[:, 0].cpu()).float().mean().item()
    # PQ on random gaussian data is lossy; require a modest but real recall.
    assert recall > 0.1, f"IVFPQ recall too low: {recall:.4f}"
    print(f"ivfpq_search OK: recall@{k}={recall:.4f}, top1={top1:.4f}")
