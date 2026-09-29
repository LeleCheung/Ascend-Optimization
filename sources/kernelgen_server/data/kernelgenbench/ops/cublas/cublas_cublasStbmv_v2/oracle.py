REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasStbmv_v2 as _baseline


def run(uplo, trans, diag, n, k, A, lda, x, incx):
    _baseline(uplo, trans, diag, n, k, A, lda, x, incx)
    return x
