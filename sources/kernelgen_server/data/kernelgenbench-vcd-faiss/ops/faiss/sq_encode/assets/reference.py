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


def sq_encode(x, vmin, vdiff):
    """Scalar-quantizer (QT_8bit, uniform) encoding.

    CUDA operator (baseline). faiss' scalar-quantizer encode path is
        faiss/gpu/impl/scan/IVFInterleaved.cuh + faiss/impl/ScalarQuantizer.cpp
            Codec8bit::encode_component  (invoked from sqEncode)
    Each component is affinely mapped into [0, 1] using the trained per-dim
    minimum (vmin) and range (vdiff), then quantized to a uint8 bucket:
        code = uint8( 255 * clamp((x - vmin) / vdiff, 0, 1) )   (truncation)

    faiss exposes ScalarQuantizer.compute_codes on CPU only, so encoding runs
    through faiss' CPU kernel and the result returns on the input device. The
    trained parameters (vmin, vdiff) are supplied as inputs and injected into
    the quantizer, so the mapping is fully determined by the inputs and a
    Triton solution can reproduce it exactly.

    Args:
        x (Tensor): vectors to encode, shape (n, d), float32.
        vmin (Tensor): per-dimension minimum, shape (d,), float32.
        vdiff (Tensor): per-dimension range (max - min), shape (d,), float32,
            strictly positive.

    Returns:
        Tensor: uint8 codes, shape (n, d).
    """
    if faiss is None:
        raise RuntimeError("sq_encode baseline requires faiss to be installed")

    import numpy as np

    n, d = x.shape
    assert vmin.shape == (d,) and vdiff.shape == (d,), (vmin.shape, vdiff.shape)

    x_np = np.ascontiguousarray(x.detach().cpu().numpy(), dtype="float32")
    trained = np.concatenate(
        [
            np.ascontiguousarray(vmin.detach().cpu().numpy(), dtype="float32"),
            np.ascontiguousarray(vdiff.detach().cpu().numpy(), dtype="float32"),
        ]
    ).astype("float32")

    sq = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
    # Inject trained params instead of sq.train(x) so the codebook is fixed.
    faiss.copy_array_to_vector(trained, sq.trained)
    codes = sq.compute_codes(x_np)  # (n, d) uint8

    return torch.from_numpy(np.ascontiguousarray(codes)).to(x.device)


if __name__ == "__main__":
    torch.manual_seed(0)

    if faiss is None:
        print("skip: sq_encode requires faiss to be installed")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def ref_sq_encode(x, vmin, vdiff):
        # Independent reference: faiss Codec8bit is a truncating uniform coder.
        xn = torch.clamp((x - vmin) / vdiff, 0.0, 1.0)
        return (255.0 * xn).to(torch.uint8)

    def sq_decode(codes, vmin, vdiff):
        # faiss reconstruction: x ~= vmin + (code + 0.5) / 255 * vdiff.
        return vmin + (codes.float() + 0.5) / 255.0 * vdiff

    n, d = 200, 64
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    vmin = x.min(dim=0).values - 1e-3
    vdiff = (x.max(dim=0).values + 1e-3) - vmin

    codes = sq_encode(x, vmin, vdiff)
    ref = ref_sq_encode(x, vmin, vdiff)

    assert codes.shape == (n, d) and codes.dtype == torch.uint8, (codes.shape, codes.dtype)
    agreement = (codes.cpu() == ref.cpu()).float().mean().item()
    assert agreement == 1.0, f"sq_encode disagrees with reference coder: {agreement:.4f}"

    # Round-trip: 8-bit uniform quantization is lossy, so check relative error.
    x_rec = sq_decode(codes, vmin, vdiff)
    rel_err = (x_rec.cpu() - x.cpu()).abs().mean() / x.cpu().abs().mean()
    assert rel_err < 0.05, f"round-trip relative error too large: {rel_err:.4f}"
    print(
        f"sq_encode OK: encoded {tuple(codes.shape)}, code agreement = {agreement:.4f}, "
        f"round-trip rel_err = {rel_err:.4f}"
    )
