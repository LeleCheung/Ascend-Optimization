REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSgemvStridedBatched as _baseline


def run(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    _baseline(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    return y
