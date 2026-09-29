REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasSdgmm as _baseline


def run(mode, m, n, A, lda, x, incx, C, ldc):
    _baseline(mode, m, n, A, lda, x, incx, C, ldc)
    return C
