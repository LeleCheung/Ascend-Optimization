REFERENCE_DEVICE = 'target'

from kernelgenbench.dataset.baseline.cublas import cublasDgemmStridedBatched as _baseline


def run(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    _baseline(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    return C
