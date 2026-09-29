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


def ivf_flat_add(xb, centroids, start_id=0):
    """faiss IndexIVFFlat vector insertion (assign-to-list then append).

    Adding vectors to an IVF index is a two-part operation: each vector is
    assigned to its nearest coarse centroid, then its raw data and its global
    id are appended to that centroid's inverted list.

    CUDA operators (baseline). On GPU the append path runs in faiss' CUDA kernels:
        faiss/gpu/impl/IVFFlat.cu::runIVFFlatAppend    -> ivfFlatAppend
        faiss/gpu/impl/IVFAppend.cu::ivfInterleavedAppend (interleaved layout)
        faiss/gpu/impl/IVFAppend.cu::ivfIndicesAppend  (per-list global ids)
        faiss/gpu/utils/DeviceUtils.cuh::runIncrementIndex -> incrementIndex
        faiss/gpu/impl/IVFBase.cu::runUpdateListPointers -> runUpdateListPointers

    Args:
        xb (Tensor):        vectors to add, shape (n, d), float32.
        centroids (Tensor): coarse centroids, shape (nlist, d), float32. Vectors
                            are assigned to the nearest centroid under L2.
        start_id (int):     global id offset of the first added vector; ids are
                            assigned start_id, start_id+1, ... in input order
                            (mirrors incrementIndex / ivfIndicesAppend).

    Returns:
        (assignments, list_lengths) where
            assignments (Tensor):  per-vector coarse-list id, shape (n,), int64.
            list_lengths (Tensor): number of vectors appended to each list,
                                   shape (nlist,), int64.
        The pair captures the observable result of the append: which list each
        vector landed in and how the inverted lists grew.

    Both outputs are torch tensors on the same device as the inputs.
    """
    if faiss is None:
        raise RuntimeError("ivf_flat_add baseline requires faiss to be installed")

    n, d = xb.shape
    nlist, d2 = centroids.shape
    assert d == d2, f"dim mismatch: xb d={d}, centroids d={d2}"

    xb = xb.contiguous().float()
    centroids = centroids.contiguous().float()

    import numpy as np

    xb_np = np.ascontiguousarray(xb.detach().cpu().numpy(), dtype="float32")
    cent_np = np.ascontiguousarray(centroids.detach().cpu().numpy(), dtype="float32")

    # Coarse quantizer holding the given centroids; assign each vector to its
    # nearest list (the "which inverted list" half of the append).
    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(cent_np)
    _, assign_np = quantizer.search(xb_np, 1)
    assign_np = assign_np.reshape(-1).astype("int64")

    # Build the real IVF index and add with explicit global ids so the inverted
    # lists are populated exactly as ivfInterleavedAppend / ivfIndicesAppend do.
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    ivf.is_trained = True  # centroids are supplied directly, no k-means needed
    ids = np.arange(start_id, start_id + n, dtype="int64")
    ivf.add_with_ids(xb_np, ids)

    lengths_np = np.array(
        [ivf.invlists.list_size(l) for l in range(nlist)], dtype="int64"
    )

    assignments = torch.from_numpy(assign_np).to(xb.device)
    list_lengths = torch.from_numpy(lengths_np).to(xb.device)
    return assignments, list_lengths


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: ivf_flat_add requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    def ref_assign(xb, centroids):
        # Independent reference: nearest-centroid assignment via cdist, and the
        # resulting per-list counts.
        dist = torch.cdist(xb.float(), centroids.float(), p=2) ** 2
        assign = torch.argmin(dist, dim=1).to(torch.int64)
        lengths = torch.bincount(assign, minlength=centroids.shape[0]).to(torch.int64)
        return assign, lengths

    n, d, nlist = 512, 64, 16
    xb = torch.randn(n, d, device=device, dtype=dtype)
    centroids = torch.randn(nlist, d, device=device, dtype=dtype)

    assign, lengths = ivf_flat_add(xb, centroids)
    assign_ref, lengths_ref = ref_assign(xb, centroids)

    assert assign.shape == (n,) and lengths.shape == (nlist,)
    assert int(lengths.sum().item()) == n, lengths.sum().item()
    agree = (assign.cpu() == assign_ref.cpu()).float().mean().item()
    torch.testing.assert_close(lengths.cpu(), lengths_ref.cpu())
    print(f"ivf_flat_add OK: {n} vectors, list-length match, assign agreement={agree:.4f}")
