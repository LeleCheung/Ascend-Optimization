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


def _torch_hamming_topk(xq, xb, k):
    """CPU/GPU torch fallback: brute-force Hamming distance + top-k.

    Popcount of the XOR of every query/database byte pair, summed over bytes.
    """
    # Byte-wise popcount lookup table (0..255).
    lut = torch.tensor(
        [bin(i).count("1") for i in range(256)],
        device=xq.device,
        dtype=torch.int32,
    )
    xq_i = xq.to(torch.int64)
    xb_i = xb.to(torch.int64)
    # xor: (nq, nb, d_bytes)
    xor = xq_i[:, None, :] ^ xb_i[None, :, :]
    dist = lut[xor].sum(dim=-1).to(torch.int32)  # (nq, nb)
    D, I = torch.topk(dist, k, dim=1, largest=False, sorted=True)
    return D.to(torch.int32), I.to(torch.int64)


def binary_knn_hamming(xq, xb, k):
    """Binary Hamming kNN: end-to-end top-k retrieval under Hamming distance.

    CUDA operator (baseline):
        faiss/gpu/impl/BinaryDistance.cu -> binaryDistanceAnySize
    faiss packs each vector into bytes, computes the Hamming distance
    (popcount of XOR) between every query and database code, and selects the
    k smallest per query. This baseline drives faiss.IndexBinaryFlat, whose
    search reproduces that kernel on CPU.

    Args:
        xq (Tensor): binary query codes, shape (nq, d_bytes), uint8. The
            underlying bit dimension is ``d_bytes * 8``.
        xb (Tensor): binary database codes, shape (nb, d_bytes), uint8.
        k (int): number of nearest neighbors to return.

    Returns:
        (D, I) where
            D (Tensor): Hamming distances to the k nearest neighbors,
                shape (nq, k), int32, ascending along dim 1.
            I (Tensor): indices of those neighbors into xb, shape (nq, k),
                int64.
        Both outputs live on the same device as the inputs.
    """
    if faiss is None:
        raise RuntimeError(
            "binary_knn_hamming baseline requires faiss to be installed"
        )

    nq, d_bytes = xq.shape
    nb, d_bytes2 = xb.shape
    assert d_bytes == d_bytes2, f"byte-dim mismatch: xq {d_bytes}, xb {d_bytes2}"

    import numpy as np

    d_bits = d_bytes * 8
    xq_np = np.ascontiguousarray(xq.detach().cpu().numpy(), dtype="uint8")
    xb_np = np.ascontiguousarray(xb.detach().cpu().numpy(), dtype="uint8")

    index = faiss.IndexBinaryFlat(d_bits)
    index.add(xb_np)
    D_np, I_np = index.search(xq_np, k)

    D = torch.from_numpy(np.ascontiguousarray(D_np)).to(torch.int32).to(xq.device)
    I = torch.from_numpy(np.ascontiguousarray(I_np)).to(torch.int64).to(xq.device)
    return D, I


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: binary_knn_hamming requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nq, nb, d_bits, k = 10, 100, 256, 5
    d_bytes = d_bits // 8
    xq = torch.randint(0, 256, (nq, d_bytes), device=device, dtype=torch.uint8)
    xb = torch.randint(0, 256, (nb, d_bytes), device=device, dtype=torch.uint8)

    D, I = binary_knn_hamming(xq, xb, k)
    # Independent torch reference: brute-force popcount(xor) + topk.
    D_ref, I_ref = _torch_hamming_topk(xq, xb, k)

    assert D.shape == (nq, k) and I.shape == (nq, k), (D.shape, I.shape)
    # Distances must match exactly.
    assert torch.equal(D.cpu(), D_ref.cpu()), "Hamming distances differ from reference"
    # Indices agree where distances are unambiguous; ties may reorder, so
    # verify the selected distance sets are identical instead.
    idx_match = (I.cpu() == I_ref.cpu()).float().mean().item()
    print(
        f"binary_knn_hamming OK: distances match exactly, "
        f"index agreement = {idx_match:.4f}"
    )
