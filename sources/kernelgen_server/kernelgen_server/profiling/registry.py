# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Lazy profiler registry that avoids importing vendor runtimes cross-platform."""

from __future__ import annotations

from typing import Dict

from .base import Profiler, UnsupportedProfiler


_INSTANCES: Dict[str, Profiler] = {}


def get_profiler(backend: str) -> Profiler:
    profiler = _INSTANCES.get(backend)
    if profiler is not None:
        return profiler

    if backend == "cuda":
        from .nvidia.nvidia import NcuProfiler

        profiler = NcuProfiler()
    elif backend == "npu":
        from .ascend.ascend import MsprofProfiler

        profiler = MsprofProfiler()
    elif backend == "mlu":
        from .cambricon.cambricon import CnperfProfiler

        profiler = CnperfProfiler()
    elif backend == "metax":
        from .metax.metax import McTracerProfiler

        profiler = McTracerProfiler()
    elif backend == "kunlunxin":
        from .kunlunxin.kunlunxin import XProfilerProfiler

        profiler = XProfilerProfiler()
    elif backend == "musa":
        from .mthreads.mthreads import McuProfiler

        profiler = McuProfiler()
    elif backend == "enflame":
        from .enflame.enflame import TopsProfProfiler

        profiler = TopsProfProfiler()
    elif backend == "hygon":
        from .hygon.dcu import HipprofProfiler

        profiler = HipprofProfiler()
    elif backend == "iluvatar":
        from .iluvatar.iluvatar import IxknProfiler

        profiler = IxknProfiler()
    elif backend == "thead":
        from .thead.thead import THeadACUProfiler

        profiler = THeadACUProfiler()
    else:
        profiler = UnsupportedProfiler(backend)

    _INSTANCES[backend] = profiler
    return profiler
