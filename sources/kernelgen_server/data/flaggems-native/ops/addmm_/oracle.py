import torch

REFERENCE_DEVICE = "target"


def _reference_result(self, mat1, mat2, *, beta, alpha):
    try:
        return torch.addmm(
            self.detach().to(torch.float64),
            mat1.detach().to(torch.float64),
            mat2.detach().to(torch.float64),
            beta=beta,
            alpha=alpha,
        )
    except RuntimeError:
        return torch.addmm(
            self.detach().to(torch.float32),
            mat1.detach().to(torch.float32),
            mat2.detach().to(torch.float32),
            beta=beta,
            alpha=alpha,
        )


def timing_run(self, mat1, mat2, *, beta=1, alpha=1):
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        return self.addmm_(mat1, mat2, beta=beta, alpha=alpha)
    finally:
        torch.set_float32_matmul_precision(precision)


def correctness_run(self, mat1, mat2, *, beta=1, alpha=1):
    result = _reference_result(self, mat1, mat2, beta=beta, alpha=alpha)
    return self.copy_(result.to(self.dtype))
