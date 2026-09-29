REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasDgemv_v2 as _baseline


def run(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    _baseline(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    return y
