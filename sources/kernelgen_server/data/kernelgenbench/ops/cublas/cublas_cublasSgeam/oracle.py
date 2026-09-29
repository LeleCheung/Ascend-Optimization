REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSgeam as _baseline


def run(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    _baseline(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    return C
