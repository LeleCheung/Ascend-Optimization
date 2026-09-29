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


def _reconstruct(codebooks, codes):
    """Additive reconstruction: sum the chosen codeword of every codebook.

    codebooks: (M, ksub, d)   codes: (n, M) int64  ->  (n, d)
    """
    M = codebooks.shape[0]
    n = codes.shape[0]
    d = codebooks.shape[2]
    out = torch.zeros(n, d, device=codebooks.device, dtype=codebooks.dtype)
    for m in range(M):
        out += codebooks[m][codes[:, m]]
    return out


def icm_encode(x, codebooks, codes_init, n_iters=4):
    """ICM (Iterated Conditional Modes) additive-quantization encoding.

    CUDA operators (baseline):
        faiss/gpu/impl/IcmEncoder.cu -> runIcmEncodeStep, runEvaluation,
                                        runCodesPerturbation, runCodesSelection
    faiss encodes each vector x as a sum of M codewords (one per codebook) by
    coordinate descent: holding all other codebook assignments fixed, it picks
    the codeword of the current codebook that minimizes the reconstruction
    error. runEvaluation scores candidate codes, runIcmEncodeStep performs one
    sweep over codebooks, runCodesSelection keeps the best assignment, and
    runCodesPerturbation escapes local minima (omitted here; deterministic
    descent is enough to guarantee monotonic error decrease).

    Args:
        x (Tensor): vectors to encode, shape (n, d), float32.
        codebooks (Tensor): additive codebooks, shape (M, ksub, d), float32.
            codebooks[m, c] is a full d-dim codeword.
        codes_init (Tensor): initial code assignment, shape (n, M), integer.
        n_iters (int): number of full coordinate-descent sweeps.

    Returns:
        Tensor: refined codes, shape (n, M), int64. Reconstructing x from these
        codes yields a strictly smaller squared error than from codes_init
        (until convergence).
    """
    M, ksub, d = codebooks.shape
    n, M2 = codes_init.shape
    assert M == M2, f"codebook count mismatch: codebooks {M}, codes {M2}"
    assert x.shape == (n, d), f"x shape {tuple(x.shape)} != ({n}, {d})"

    x = x.float()
    codebooks = codebooks.float()
    codes = codes_init.to(torch.int64).clone()

    # Running full reconstruction; updated incrementally as codes change.
    recon = _reconstruct(codebooks, codes)

    for _ in range(n_iters):
        for m in range(M):
            # Residual with codebook m's current contribution removed.
            contrib_m = codebooks[m][codes[:, m]]            # (n, d)
            residual = x - (recon - contrib_m)               # (n, d)
            # Squared distance from residual to every codeword of codebook m.
            # (n, ksub, d) -> (n, ksub)
            dist = ((residual[:, None, :] - codebooks[m][None, :, :]) ** 2).sum(dim=-1)
            best = dist.argmin(dim=1)                         # (n,)
            new_contrib = codebooks[m][best]                 # (n, d)
            # Commit: swap m's contribution in the running reconstruction.
            recon = recon - contrib_m + new_contrib
            codes[:, m] = best

    return codes


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n, d, M, ksub = 100, 64, 8, 256
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    # Small-scale codebooks so a random init leaves clear room to improve.
    codebooks = torch.randn(M, ksub, d, device=device, dtype=torch.float32) * 0.3
    codes_init = torch.randint(0, ksub, (n, M), device=device, dtype=torch.int64)

    def sq_error(codes):
        recon = _reconstruct(codebooks, codes)
        return ((x - recon) ** 2).sum(dim=-1).mean().item()

    err_init = sq_error(codes_init)
    codes = icm_encode(x, codebooks, codes_init, n_iters=4)
    err_final = sq_error(codes)

    assert codes.shape == (n, M), codes.shape
    # Real ICM must monotonically reduce reconstruction error.
    assert err_final < err_init, (
        f"ICM did not reduce error: init={err_init:.4f}, final={err_final:.4f}"
    )
    print(
        f"icm_encode OK: reconstruction error {err_init:.4f} -> {err_final:.4f} "
        f"(monotonic decrease)"
    )
