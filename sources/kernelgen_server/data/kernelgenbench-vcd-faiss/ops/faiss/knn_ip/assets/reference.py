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


def knn_ip(xq, xb, k):
    """faiss brute-force k-nearest-neighbor search under the inner-product metric.

    CUDA operator (baseline). Runs in faiss' C++/CUDA kernels:
        faiss/gpu/GpuDistance.cu::bfKnn -> runIPDistance (GEMM)
            -> l2SelectMinK (faiss/gpu/impl/L2Select.cu), max-heap variant.

    Under inner product, "nearest" means largest dot product, so results are
    sorted in descending order of similarity.

    Args:
        xq (Tensor): query vectors, shape (nq, d), float32.
        xb (Tensor): database vectors, shape (nb, d), float32.
        k (int):     number of neighbors to return.

    Returns:
        (D, I) where
            D (Tensor): inner-product similarities, shape (nq, k), float32,
                        descending along dim 1.
            I (Tensor): neighbor indices into xb, shape (nq, k), int64.
    """
    if faiss is None:
        raise RuntimeError("knn_ip baseline requires faiss to be installed")

    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2, f"dim mismatch: xq d={d}, xb d={d2}"

    xq = xq.contiguous().float()
    xb = xb.contiguous().float()

    use_gpu = (
        xq.is_cuda
        and hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "knn_gpu")
    )

    if use_gpu:
        res = _get_gpu_res()
        D, I = faiss.knn_gpu(res, xq, xb, k, metric=faiss.METRIC_INNER_PRODUCT)
        return D.contiguous(), I.to(torch.int64).contiguous()

    import numpy as np

    xq_np = np.ascontiguousarray(xq.detach().cpu().numpy(), dtype="float32")
    xb_np = np.ascontiguousarray(xb.detach().cpu().numpy(), dtype="float32")
    D_np, I_np = faiss.knn(xq_np, xb_np, k, metric=faiss.METRIC_INNER_PRODUCT)
    D = torch.from_numpy(np.ascontiguousarray(D_np)).to(xq.device)
    I = torch.from_numpy(np.ascontiguousarray(I_np)).to(xq.device).to(torch.int64)
    return D, I


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: knn_ip requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    def ref_knn_ip(xq, xb, k):
        sim = xq.float() @ xb.float().t()
        D, I = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        return D, I.to(torch.int64)

    nq, nb, d, k = 128, 2048, 64, 10
    xq = torch.randn(nq, d, device=device, dtype=dtype)
    xb = torch.randn(nb, d, device=device, dtype=dtype)

    D, I = knn_ip(xq, xb, k)
    D_ref, I_ref = ref_knn_ip(xq, xb, k)

    assert D.shape == (nq, k) and I.shape == (nq, k), (D.shape, I.shape)
    torch.testing.assert_close(D.cpu(), D_ref.cpu(), rtol=1e-3, atol=1e-3)
    idx_match = (I.cpu() == I_ref.cpu()).float().mean().item()
    print(f"knn_ip OK: similarities match, index agreement = {idx_match:.4f}")
