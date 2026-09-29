REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSsyrk_v2 as _baseline


def run(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    _baseline(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    return C
