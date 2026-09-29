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


def calc_residual(x, centroids, assignments):
    """Per-vector residual against an assigned centroid.

    CUDA operator (baseline):
        faiss/gpu/impl/VectorResidual.cu::calcResidual
    For each vector it subtracts the centroid it was assigned to. This is used
    by IVF/IVFPQ before quantizing the residual of each list entry.

    This op is elementwise gather + subtract, so it is computed exactly in
    torch on the input device (no faiss round-trip needed); the baseline is
    itself the exact reference.

    Args:
        x (Tensor): vectors, shape (n, d), float.
        centroids (Tensor): cluster centers, shape (k, d), float.
        assignments (Tensor): assigned cluster id per vector, shape (n,), int.

    Returns:
        Tensor: residuals x - centroids[assignments], shape (n, d), float.
    """
    assert x.shape[1] == centroids.shape[1], (x.shape, centroids.shape)
    assert assignments.shape[0] == x.shape[0], (assignments.shape, x.shape)
    return x - centroids[assignments.long()]


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def ref_calc_residual(x, centroids, assignments):
        # Independent reference via an explicit gather loop (no fancy indexing).
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = x[i] - centroids[int(assignments[i])]
        return out

    n, k, d = 100, 10, 64
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    centroids = torch.randn(k, d, device=device, dtype=torch.float32)
    assignments = torch.randint(0, k, (n,), device=device, dtype=torch.int64)

    res = calc_residual(x, centroids, assignments)
    ref = ref_calc_residual(x, centroids, assignments)

    assert res.shape == (n, d), res.shape
    torch.testing.assert_close(res.cpu(), ref.cpu(), rtol=0, atol=0)
    print(f"calc_residual OK: {tuple(res.shape)}, exact match with reference")
