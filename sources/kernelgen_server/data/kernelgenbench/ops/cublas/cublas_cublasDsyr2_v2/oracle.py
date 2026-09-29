REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasDsyr2_v2 as _baseline


def run(uplo, n, alpha, x, incx, y, incy, A, lda):
    _baseline(uplo, n, alpha, x, incx, y, incy, A, lda)
    return A
