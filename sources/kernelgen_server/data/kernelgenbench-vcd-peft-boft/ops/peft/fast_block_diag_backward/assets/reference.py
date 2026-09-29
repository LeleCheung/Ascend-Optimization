# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained PEFT BOFT Fast Block Diagonal CUDA backward baseline.

The CUDA indexing and launch configuration come from PEFT commit
``004fd0369e770dc857b895050f65bb40f7bca89c``. ``Tensor.scalar_type()`` replaces
the removed ``Tensor.type()`` dispatch API so the extracted source builds with
current PyTorch. No PEFT installation is required on the evaluator.
"""

from functools import lru_cache

import torch


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> backward_fast_block_diag_cuda(
    at::Tensor grad_output,
    at::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward_fast_block_diag_cuda, "FAST BLOCK DIAG backward (CUDA)");
}
"""


_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void backward_fast_block_diag_cuda_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* grad_input,
    int z,
    int n,
    int b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= z * n * b * b) {
        return;
    }
    const int zi = i / (n * b * b);
    const int ni = (i % (n * b * b)) / (b * b);
    const int x = ((i % (n * b * b)) % (b * b)) / b;
    const int y = ((i % (n * b * b)) % (b * b)) % b;
    grad_input[zi * n * b * b + ni * b * b + x * b + y] =
        grad_output[zi * n * b * n * b + (ni * b + x) * n * b + ni * b + y];
}

std::vector<at::Tensor> backward_fast_block_diag_cuda(
    at::Tensor grad_output,
    at::Tensor input) {
    const auto z = input.size(0);
    const auto n = input.size(1);
    const auto b = input.size(2);
    const int threads = 512;
    const dim3 blocks((z * n * b * b - 1) / threads + 1);
    auto grad_input = at::zeros_like(input);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(), "backward_fast_block_diag", ([&] {
            backward_fast_block_diag_cuda_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(), z, n, b);
        }));

    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
    return {grad_input};
}
"""


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="kernelgenbench_peft_fbd_backward_cuda",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        extra_cuda_cflags=["-lineinfo"],
        verbose=False,
    )


def fast_block_diag_backward(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """Gather the diagonal blocks of ``grad_output`` into ``input``'s shape."""
    return _extension().backward(grad_output, input)[0]
