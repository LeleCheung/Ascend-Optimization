REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasDasum_v2 as _baseline


def run(n, x, incx, result):
    _baseline(n, x, incx, result)
    return result
