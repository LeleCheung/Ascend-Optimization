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


def ivf_flat_search(xq, xb, k, nlist=100, nprobe=10):
    """faiss IndexIVFFlat approximate nearest-neighbor search (L2 metric).

    An inverted-file (IVF) index partitions the database into ``nlist`` Voronoi
    cells via a coarse quantizer. A query is compared only against the vectors
    in its ``nprobe`` closest cells, then the top-k are selected.

    CUDA operators (baseline). On GPU the search runs in faiss' CUDA kernels:
        faiss/gpu/impl/IVFFlatScan.cu::ivfFlatScan           (flat per-list scan)
        faiss/gpu/impl/IVFInterleaved.cu::ivfInterleavedScan (interleaved codes)
        faiss/gpu/impl/IVFInterleaved.cuh::ivfInterleavedScan2
        faiss/gpu/impl/IVFUtilsSelect1.cu::pass1SelectLists  (per-list top-k)
        faiss/gpu/impl/IVFUtilsSelect2.cu::pass2SelectLists  (merge to final k)

    Args:
        xq (Tensor):  query vectors, shape (nq, d), float32.
        xb (Tensor):  database vectors, shape (nb, d), float32.
        k (int):      number of nearest neighbors to return.
        nlist (int):  number of IVF cells (coarse centroids).
        nprobe (int): number of cells visited per query.

    Returns:
        (D, I) where
            D (Tensor): squared-L2 distances, shape (nq, k), float32,
                        ascending along dim 1.
            I (Tensor): neighbor indices into xb, shape (nq, k), int64.
                        Padding value -1 may appear when fewer than k
                        candidates are found in the probed cells.

    Both outputs are torch tensors on the same device as the inputs.
    """
    if faiss is None:
        raise RuntimeError("ivf_flat_search baseline requires faiss to be installed")

    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2, f"dim mismatch: xq d={d}, xb d={d2}"
    nlist = min(nlist, nb)
    nprobe = min(nprobe, nlist)

    xq = xq.contiguous().float()
    xb = xb.contiguous().float()

    use_gpu = (
        xq.is_cuda
        and hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "GpuIndexIVFFlat")
    )

    if use_gpu:
        # GPU path -> exercises ivfFlatScan / ivfInterleavedScan / pass{1,2}SelectLists.
        res = _get_gpu_res()
        cfg = faiss.GpuIndexIVFFlatConfig()
        index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, cfg)
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
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
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
        print("skip: ivf_flat_search requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    def ref_bruteforce_knn(xq, xb, k):
        # Exact reference: full pairwise squared-L2 + topk. IVF is approximate,
        # so we score it against exact search by recall rather than equality.
        dist = torch.cdist(xq.float(), xb.float(), p=2) ** 2
        _, I = torch.topk(dist, k, dim=1, largest=False, sorted=True)
        return I.to(torch.int64)

    nq, nb, d, k = 64, 4096, 64, 10
    nlist, nprobe = 64, 16
    xq = torch.randn(nq, d, device=device, dtype=dtype)
    xb = torch.randn(nb, d, device=device, dtype=dtype)

    D, I = ivf_flat_search(xq, xb, k, nlist=nlist, nprobe=nprobe)
    I_exact = ref_bruteforce_knn(xq, xb, k)

    assert D.shape == (nq, k) and I.shape == (nq, k), (D.shape, I.shape)
    # Recall@k of the IVF neighbor set against exact top-k.
    hits = 0
    for q in range(nq):
        approx = set(I[q].cpu().tolist()) - {-1}
        exact = set(I_exact[q].cpu().tolist())
        hits += len(approx & exact)
    recall = hits / (nq * k)
    top1 = (I[:, 0].cpu() == I_exact[:, 0].cpu()).float().mean().item()
    assert recall > 0.5, f"IVF recall too low: {recall:.4f}"
    print(f"ivf_flat_search OK: recall@{k}={recall:.4f}, top1={top1:.4f}")
