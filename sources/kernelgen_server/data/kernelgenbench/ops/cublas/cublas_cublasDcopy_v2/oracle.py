REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasDcopy_v2 as _baseline


def run(n, x, incx, y, incy):
    _baseline(n, x, incx, y, incy)
    return y
