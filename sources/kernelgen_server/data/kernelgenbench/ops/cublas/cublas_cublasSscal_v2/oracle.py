REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSscal_v2 as _baseline


def run(n, alpha, x, incx):
    _baseline(n, alpha, x, incx)
    return x
