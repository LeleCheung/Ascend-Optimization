"""In-process profiling capture boundaries for the Native runner.

The outer vendor tool starts the candidate process with capture disabled.  The
candidate process then opens the backend-specific range only after inputs,
compilation, warmup, and the first synchronization have completed.
"""

from __future__ import annotations

import ctypes
import os
import time
from contextlib import contextmanager
from typing import Callable, Iterator


class ProfilerControlError(RuntimeError):
    """A vendor runtime could not open or close its profiling range."""


def prepare_profile_compiler(backend: str) -> None:
    """Apply the same compiler preparation before Native or pytest imports."""

    if backend != "enflame":
        return
    if os.environ.get("TRITON_DISABLE_LINE_INFO", "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }:
        return
    from .enflame.enflame_line_info import enable_enflame_line_info_compat

    enable_enflame_line_info_compat()


def _runtime_control(library: str, function_name: str) -> None:
    try:
        runtime = ctypes.CDLL(library)
    except OSError as exc:
        raise ProfilerControlError(f"could not load {library}: {exc}") from exc
    try:
        function: Callable[[], int] = getattr(runtime, function_name)
    except AttributeError as exc:
        raise ProfilerControlError(
            f"{library} does not export {function_name}"
        ) from exc
    function.argtypes = []
    function.restype = ctypes.c_int
    status = int(function())
    if status != 0:
        raise ProfilerControlError(f"{function_name} failed with status {status}")


def _cuda_control(start: bool) -> None:
    import torch

    cudart = torch.cuda.cudart()
    function_name = "cudaProfilerStart" if start else "cudaProfilerStop"
    result = getattr(cudart, function_name)()
    # PyTorch CUDA bindings normally return 0 or a one-item tuple containing 0.
    status = result[0] if isinstance(result, tuple) and result else result
    if status not in (None, 0):
        raise ProfilerControlError(f"{function_name} failed with status {status}")


def _start(backend: str) -> int | None:
    if backend in {"cuda", "iluvatar"}:
        _cuda_control(True)
    elif backend == "mlu":
        _runtime_control("libcnrt.so", "cnrtProfilerStart")
    elif backend == "metax":
        _runtime_control("libmcruntime.so", "mcProfilerStart")
    elif backend == "enflame":
        _runtime_control("libtopsrt.so", "topsProfilerStart")
    elif backend == "thead":
        _runtime_control("libhggcrt1.so", "hggcProfilerStart")
    elif backend == "kunlunxin":
        start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        print(f"KGS_PROFILE_START_NS={start_ns}", flush=True)
        return start_ns
    # Ascend, Hygon, and MThreads use vendor collection modes that do not expose
    # a verified in-process start/stop API in the supported toolchain.
    return None


def _stop(backend: str, start_ns: int | None) -> None:
    if backend in {"cuda", "iluvatar"}:
        _cuda_control(False)
    elif backend == "mlu":
        _runtime_control("libcnrt.so", "cnrtProfilerStop")
    elif backend == "metax":
        _runtime_control("libmcruntime.so", "mcProfilerStop")
    elif backend == "enflame":
        _runtime_control("libtopsrt.so", "topsProfilerStop")
    elif backend == "thead":
        _runtime_control("libhggcrt1.so", "hggcProfilerStop")
    elif backend == "kunlunxin" and start_ns is not None:
        end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        print(f"KGS_PROFILE_END_NS={end_ns}", flush=True)
        print(
            f"KGS_PROFILE_WALL_TIME_US={(end_ns - start_ns) / 1000.0:.3f}",
            flush=True,
        )


@contextmanager
def capture_scope(backend: str) -> Iterator[None]:
    """Open the fixed Native capture boundary for one logical backend."""

    start_ns = _start(backend)
    try:
        yield
    finally:
        _stop(backend, start_ns)


__all__ = ["ProfilerControlError", "capture_scope", "prepare_profile_compiler"]
