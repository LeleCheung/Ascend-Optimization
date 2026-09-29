"""Reference oracle for the avg_pool3d operator.

The reference is the ATen op ``torch.ops.aten.avg_pool3d`` exercised by the
FlagGems accuracy pytest (tests/test_avg_pool3d.py) and the benchmark
(benchmark/test_avg_pool3d_perf.py).

- ``timing_run`` reproduces the benchmark Torch baseline: the op applied to the
  materialized input at its original dtype.
- ``correctness_run`` reproduces the accuracy-pytest reference: the input is
  upcast to float64 (``to_reference(..., upcast=True)``), the op is evaluated on
  the upcast input, and the result is cast back to the original input dtype,
  mirroring the ``ref.to(dtype)`` step of ``gems_assert_close``.
"""

import torch

REFERENCE_DEVICE = "target"


def timing_run(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return torch.ops.aten.avg_pool3d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


def correctness_run(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    ref_inp = input.to(torch.float64)
    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    return ref_out.to(input.dtype)
