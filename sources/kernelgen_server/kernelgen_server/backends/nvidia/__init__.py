# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA CUDA device backend (H800, A100, H100, etc.)."""

from kernelgen_server.backends.nvidia.device import CudaDevice

__all__ = ["CudaDevice"]
