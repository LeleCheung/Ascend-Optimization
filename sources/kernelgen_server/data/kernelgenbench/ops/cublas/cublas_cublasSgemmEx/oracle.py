REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSgemmEx as _baseline


def run(transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc):
    _baseline(transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    return C
