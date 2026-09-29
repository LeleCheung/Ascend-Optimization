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
        import faiss.contrib.torch_utils  # noqa: F401  (enables torch tensor I/O)
except (ModuleNotFoundError, ImportError):
    faiss = None

# One reusable GPU resource handle (allocating it per call is expensive).
_GPU_RES = None


def _get_gpu_res():
    global _GPU_RES
    if _GPU_RES is None:
        _GPU_RES = faiss.StandardGpuResources()
    return _GPU_RES


def _canonicalize(lims, D, I, nq, device):
    """Sort each query's neighbor block by ascending database index so the
    ragged output is deterministic. faiss returns range-search matches in an
    unspecified order; sorting by index gives a stable canonical form that is
    robust to the tiny float differences that reorder near-equal distances
    (a distance-based sort is fragile exactly at the radius boundary).
    Returns torch tensors (lims int64, D float32, I int64)."""
    lims_t = torch.as_tensor(lims, dtype=torch.int64)
    D_out = torch.empty(int(lims_t[-1]), dtype=torch.float32)
    I_out = torch.empty(int(lims_t[-1]), dtype=torch.int64)
    D_t = torch.as_tensor(D, dtype=torch.float32)
    I_t = torch.as_tensor(I, dtype=torch.int64)
    for q in range(nq):
        s, e = int(lims_t[q]), int(lims_t[q + 1])
        if e <= s:
            continue
        order = torch.argsort(I_t[s:e], stable=True)
        D_out[s:e] = D_t[s:e][order]
        I_out[s:e] = I_t[s:e][order]
    return (
        lims_t.to(device),
        D_out.to(device),
        I_out.to(device),
    )


def range_search_l2(xq, xb, radius):
    """faiss brute-force range search under the (squared) L2 metric.

    Returns every database vector whose squared-L2 distance to a query is
    strictly less than ``radius`` (faiss uses a strict ``<`` threshold).

    CUDA operator (baseline). The heavy lifting runs in faiss' C++/CUDA kernels:
        faiss/gpu/GpuDistance.cu::bfKnn / runL2Distance
            -> l2NormRowMajor (faiss/gpu/impl/L2Norm.cu) + GEMM cross term
            -> getResultLengths (faiss/gpu/impl/RemapIndices / IVFUtils.cu),
               which produces the per-query result counts that become the
               ``lims`` prefix-sum offsets of the ragged output.

    Args:
        xq (Tensor):   query vectors, shape (nq, d), dtype float32.
        xb (Tensor):   database vectors, shape (nb, d), dtype float32.
        radius (float): squared-L2 cutoff; neighbors with dist < radius are kept.

    Returns:
        (lims, D, I) where
            lims (Tensor): int64, shape (nq + 1,). lims[q]:lims[q+1] slices the
                           flat result arrays for query q. lims[-1] == total.
            I (Tensor):    database indices of all matches, shape (total,),
                           int64, ascending within each query block.
            D (Tensor):    squared-L2 distances of all matches, shape (total,),
                           float32, ordered to match I.

    All outputs are torch tensors on the same device as the inputs.
    """
    if faiss is None:
        raise RuntimeError(
            "range_search_l2 baseline requires faiss to be installed"
        )

    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2, f"dim mismatch: xq d={d}, xb d={d2}"
    radius = float(radius)

    xq = xq.contiguous().float()
    xb = xb.contiguous().float()

    use_gpu = (
        xq.is_cuda
        and hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "GpuIndexFlatL2")
    )

    if use_gpu:
        # GPU path -> exercises the CUDA distance + getResultLengths kernels.
        res = _get_gpu_res()
        index = faiss.GpuIndexFlatL2(res, d)
        index.add(xb)
        lims, D, I = index.range_search(xq, radius)
        return _canonicalize(lims, D, I, nq, xq.device)

    # CPU path: faiss.IndexFlatL2 + range_search (numpy round-trip).
    import numpy as np

    xq_np = np.ascontiguousarray(xq.detach().cpu().numpy(), dtype="float32")
    xb_np = np.ascontiguousarray(xb.detach().cpu().numpy(), dtype="float32")
    index = faiss.IndexFlatL2(d)
    index.add(xb_np)
    lims, D, I = index.range_search(xq_np, radius)
    return _canonicalize(lims, D, I, nq, xq.device)


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: range_search_l2 requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    def ref_range_search_l2(xq, xb, radius):
        # Independent reference: full pairwise squared-L2 + strict threshold,
        # results sorted by ascending database index within each query.
        dist = torch.cdist(xq.float(), xb.float(), p=2) ** 2
        nq = xq.shape[0]
        counts = []
        D_parts, I_parts = [], []
        for q in range(nq):
            mask = dist[q] < radius
            idx = torch.nonzero(mask, as_tuple=False).flatten()  # already ascending
            D_parts.append(dist[q][idx])
            I_parts.append(idx.to(torch.int64))
            counts.append(idx.numel())
        lims = torch.zeros(nq + 1, dtype=torch.int64)
        lims[1:] = torch.tensor(counts, dtype=torch.int64).cumsum(0)
        D = torch.cat(D_parts) if D_parts else torch.empty(0)
        I = torch.cat(I_parts) if I_parts else torch.empty(0, dtype=torch.int64)
        return lims, D, I

    nq, nb, d = 128, 2048, 64
    # Pick a radius that keeps a moderate, non-trivial number of neighbors.
    radius = float(2.0 * d)
    xq = torch.randn(nq, d, device=device, dtype=dtype)
    xb = torch.randn(nb, d, device=device, dtype=dtype)

    lims, D, I = range_search_l2(xq, xb, radius)
    lims_ref, D_ref, I_ref = ref_range_search_l2(xq, xb, radius)

    assert lims.shape == (nq + 1,), lims.shape
    assert torch.equal(lims.cpu(), lims_ref.cpu()), "per-query result counts differ"
    torch.testing.assert_close(D.cpu(), D_ref.cpu(), rtol=1e-4, atol=1e-4)
    assert torch.equal(I.cpu(), I_ref.cpu()), "neighbor indices differ from reference"
    total = int(lims[-1].item())
    print(f"range_search_l2 OK: {total} matches, lims + distances + indices match")
