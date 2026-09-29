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


def pca_transform(x, pca_matrix):
    """PCA dimensionality-reduction transform: x_reduced = x @ pca_matrix.T.

    CUDA operators (baseline). The projection itself is a matmul, but the
    faiss PCA train/apply path exercises a family of transpose/reduction
    primitives that this algorithm absorbs:
        faiss/gpu/impl/Transpose.cu      -> transposeAny, transposeOuter
        faiss/gpu/impl/L2Norm.cu / reduce-> sumAlongRows, sumAlongColumns
        faiss/gpu/impl/BroadcastSum.cu   -> assignAlongColumns
        faiss/gpu/impl/DistanceUtils.cu  -> runNormAddition
    faiss uses these when centering the data (mean subtraction along rows /
    columns), transposing the covariance eigenbasis into the projection
    matrix, and adding per-row norms during whitening. Once the PCAMatrix is
    trained the apply step reduces to a single dense matrix multiply, which is
    what this baseline computes with torch for backend portability.

    Args:
        x (Tensor): input vectors, shape (n, d_in), float32.
        pca_matrix (Tensor): PCA projection matrix, shape (d_out, d_in),
            float32. Row i is the i-th principal-component direction.

    Returns:
        Tensor: transformed vectors, shape (n, d_out), float32, on the same
        device as ``x``.
    """
    n, d_in = x.shape
    d_out, d_in2 = pca_matrix.shape
    assert d_in == d_in2, f"dim mismatch: x d_in={d_in}, pca d_in={d_in2}"
    return x.float() @ pca_matrix.float().t()


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import faiss
        import numpy as np
    except (ModuleNotFoundError, ImportError):
        faiss = None

    n, d_in, d_out = 512, 64, 32
    x = torch.randn(n, d_in, device=device, dtype=torch.float32)

    if faiss is not None:
        # Train a real faiss PCAMatrix, then extract its projection matrix and
        # apply it two ways: faiss.apply (C++ path) and torch matmul (baseline).
        x_np = np.ascontiguousarray(x.detach().cpu().numpy(), dtype="float32")
        pca = faiss.PCAMatrix(d_in, d_out)
        pca.train(x_np)

        # Pull the trained projection matrix (row-major (d_out, d_in)) + bias.
        A = faiss.vector_to_array(pca.A).reshape(d_out, d_in)
        b = faiss.vector_to_array(pca.b)
        pca_matrix = torch.from_numpy(np.ascontiguousarray(A)).to(device)

        # faiss reference: full trained apply (includes the mean/bias term).
        y_faiss = torch.from_numpy(pca.apply(x_np)).to(device)
        # baseline matmul + bias reproduces faiss.apply.
        y_ref = pca_transform(x, pca_matrix) + torch.from_numpy(b).to(device)

        assert y_ref.shape == (n, d_out), y_ref.shape
        torch.testing.assert_close(y_ref.cpu(), y_faiss.cpu(), rtol=1e-4, atol=1e-4)
        print("pca_transform OK: torch matmul reproduces faiss.PCAMatrix.apply")
    else:
        # No faiss: verify against an explicit torch projection.
        pca_matrix = torch.randn(d_out, d_in, device=device, dtype=torch.float32)
        y = pca_transform(x, pca_matrix)
        y_ref = torch.einsum("nd,kd->nk", x, pca_matrix)
        assert y.shape == (n, d_out), y.shape
        torch.testing.assert_close(y.cpu(), y_ref.cpu(), rtol=1e-4, atol=1e-5)
        print("pca_transform OK (no faiss): matmul(x, pca_matrix.T) verified")
