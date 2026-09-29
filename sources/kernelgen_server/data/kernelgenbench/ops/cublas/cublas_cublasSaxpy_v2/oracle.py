REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSaxpy_v2 as _baseline


def run(n, alpha, x, incx, y, incy):
    _baseline(n, alpha, x, incx, y, incy)
    return y
