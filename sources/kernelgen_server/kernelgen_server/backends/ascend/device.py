# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU device implementation."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import torch

from kernelgen_server.runtime.device import (
    Device,
    DeviceInfo,
    clean_status_value,
    device_info_from_properties,
)


def _version_from_info_file(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip() == "Version":
            return clean_status_value(value)
    return ""


@lru_cache(maxsize=None)
def _compiler_architecture(soc_name: str) -> str:
    """Resolve the compiler ISA from CANN's installed SoC database."""
    if not soc_name:
        return ""
    try:
        import tbe.common.platform as tbe_platform

        tbe_platform.set_current_compile_soc_info(soc_name)
        compiler_arch_key = getattr(tbe_platform, "COMPILER_ARCH", "")
        if not compiler_arch_key:
            return ""
        return clean_status_value(tbe_platform.get_soc_spec(compiler_arch_key))
    except Exception:
        return ""


class NpuDevice(Device):
    """Ascend NPU backend.

    Timing has three modes:
    - ``"triton"`` (default): ``triton.testing.do_bench`` inherited from Device.
    - ``"profiler"``: hardware profiler via
      ``kernelgen_server.backends.ascend.profiler.profiler_npu`` (clears L2,
      dsl="triton_ascend"); returns the profiler's *average* microseconds.
      Invalid or missing profiler output is an error and never falls back to
      another timing method.
    - ``"walltime"``: ``time.perf_counter`` over ``num_trials`` rounds; returns
      the *median* — aligned with the CUDA/wall-clock statistic and akg
      ``run_bench`` methodology.
    """

    backend = "npu"

    #: env var used to propagate perf_mode into spawned worker subprocesses
    #: (module-level ``configure_device`` state does NOT cross a spawn boundary).
    PERF_MODE_ENV = "KGS_NPU_PERF_MODE"

    def __init__(self, perf_mode: Optional[str] = None, num_trials: int = 3):
        import os

        if perf_mode is None:
            perf_mode = os.environ.get(self.PERF_MODE_ENV, "triton")
        if perf_mode not in ("profiler", "walltime", "triton"):
            raise ValueError(
                f"perf_mode must be 'profiler', 'walltime', or 'triton', got {perf_mode!r}"
            )
        self.perf_mode = perf_mode
        self.num_trials = num_trials

    def _npu(self):
        # Lazy import: torch_npu registers the npu device and has heavy side
        # effects (device acquisition), so import only when actually used.
        import torch_npu  # noqa: F401

        return torch.npu

    def set_device(self, device: str) -> None:
        self._npu().set_device(self.parse_index(device))

    def synchronize(self, device: str) -> None:
        self._npu().synchronize()

    def empty_cache(self) -> None:
        self._npu().empty_cache()

    def is_available(self) -> bool:
        try:
            return self._npu().is_available()
        except Exception:
            return False

    def list_devices(self) -> List[str]:
        return [f"npu:{i}" for i in range(self._npu().device_count())]

    def device_name(self, device: str) -> str:
        try:
            return self._npu().get_device_name(self.parse_index(device))
        except Exception:
            return ""

    def time(self, fn, args, warmup, iters, device, grad_to_none=None) -> float:
        # `warmup` / `iters` are ms budgets (see Device.time). The triton path hands
        # them straight to do_bench (which derives counts). The walltime / profiler
        # loops take counts, so convert the budget to counts first — this is what
        # keeps all three modes honoring the same ms budget as FlagGems.
        if self.perf_mode == "profiler":
            return self._time_profiler(fn, args, warmup, iters, device)
        elif self.perf_mode == "walltime":
            return self._time_walltime(fn, args, warmup, iters, device)
        else:
            # "triton" or any other mode → fall back to do_bench (inherited, ms).
            return super().time(fn, args, warmup, iters, device, grad_to_none=grad_to_none)

    def _time_profiler(self, fn, args, warmup_ms, rep_ms, device) -> float:
        from kernelgen_server.backends.ascend.profiler import profiler_npu

        # Pre-compile once (keep Triton JIT out of the timed region).
        fn(*args)
        self.synchronize(device)
        warmup_n, active_n = self.budget_to_counts(fn, args, warmup_ms, rep_ms, device)
        avg_us = profiler_npu(
            lambda: fn(*args),
            warmup=warmup_n,
            active=active_n,
            clear_l2_cache=True,
            dsl="triton_ascend",
            suppress_warnings=True,
        )
        if not math.isfinite(avg_us) or avg_us <= 0:
            raise RuntimeError(
                "Ascend profiler produced invalid timing "
                f"({avg_us!r} us); walltime fallback is disabled"
            )
        return avg_us / 1000.0  # us -> ms

    def _time_walltime(self, fn, args, warmup_ms, rep_ms, device) -> float:
        import time as _time

        warmup_n, iters_n = self.budget_to_counts(fn, args, warmup_ms, rep_ms, device)

        for _ in range(warmup_n):
            fn(*args)
        self.synchronize(device)

        trial_times: List[float] = []
        for _ in range(self.num_trials):
            self.synchronize(device)
            t0 = _time.perf_counter()
            for _ in range(iters_n):
                fn(*args)
            self.synchronize(device)
            trial_times.append((_time.perf_counter() - t0) / iters_n)

        trial_times.sort()
        return trial_times[len(trial_times) // 2] * 1000.0  # s -> ms (median)

    def count_devices_safe(self) -> int:
        """NPU: subprocess torch_npu to count without importing in main process."""
        import subprocess as _sp
        import sys as _sys

        try:
            result = _sp.run(
                [_sys.executable, "-c", "import torch, torch_npu; print(torch.npu.device_count())"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception:
            return 0

    @property
    def default_target_hardware(self) -> List[str]:
        return ["Ascend910B"]

    @property
    def default_entry_point(self) -> str:
        return "main.py::run"

    def get_device_info(self, device: str) -> DeviceInfo:
        idx = self.parse_index(device)
        try:
            props = self._npu().get_device_properties(idx)
            cores = getattr(props, "cube_core_num", 0) or 0
            if not cores:
                cores = getattr(props, "multi_processor_count", 0) or 0
            architecture = getattr(props, "gcnArchName", "") or ""
            return device_info_from_properties(
                idx,
                self.device_name(device),
                props,
                architecture=str(architecture),
                num_sm=int(cores),
            )
        except Exception:
            return DeviceInfo(device_id=idx, name=self.device_name(device))
    def status_metadata(self) -> dict[str, object]:
        # set_env.sh may select a toolkit outside the system default. A missing
        # selected version file is unknown, not permission to report another
        # installed CANN's version.
        runtime_root = (
            os.environ.get("ASCEND_HOME_PATH")
            or os.environ.get("ASCEND_TOOLKIT_HOME")
            or "/usr/local/Ascend/ascend-toolkit/latest"
        )
        runtime_path = Path(runtime_root) / "share/info/runtime/version.info"
        driver_path = Path("/usr/local/Ascend/driver/version.info")
        runtime_version = _version_from_info_file(runtime_path)
        driver_version = _version_from_info_file(driver_path)
        architecture = _compiler_architecture(self.device_name("npu:0"))
        sources = {}
        if architecture:
            sources["target.architecture"] = (
                "tbe.common.platform.get_soc_spec(COMPILER_ARCH)"
            )
        if runtime_version:
            sources["software.runtime_version"] = str(runtime_path)
        if driver_version:
            sources["software.driver_version"] = str(driver_path)
        return {
            "architecture": architecture,
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "sources": sources,
        }
