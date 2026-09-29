REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSgemm_v2 as _baseline


def run(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    _baseline(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    return C
