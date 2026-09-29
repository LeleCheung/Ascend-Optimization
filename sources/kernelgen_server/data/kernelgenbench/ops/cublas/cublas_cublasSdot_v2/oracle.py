REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSdot_v2 as _baseline


def run(n, x, incx, y, incy, result):
    _baseline(n, x, incx, y, incy, result)
    return result
