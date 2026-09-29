REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasDsbmv_v2 as _baseline


def run(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    _baseline(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    return y
