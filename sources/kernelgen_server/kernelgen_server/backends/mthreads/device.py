# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Moore Threads MUSA device implementation (MTT S5000, S4000)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from kernelgen_server.runtime.device import (
    Device,
    DeviceInfo,
    device_info_from_properties,
    run_status_command,
    status_value_from_label,
    torch_runtime_version,
)


class MusaDevice(Device):
    """Moore Threads MUSA backend.

    - Device API: torch.musa (via torch_musa package)
    - Query cmd: mthreads-gmi
    - Capabilities: fp64=False, bf16=True, int64=True
    - Default timing: Triton ``do_bench``
    """

    backend = "musa"

    def _musa(self):
        import torch_musa  # noqa: F401
        return torch.musa

    def set_device(self, device: str) -> None:
        self._musa().set_device(self.parse_index(device))

    def synchronize(self, device: str) -> None:
        self._musa().synchronize()

    def empty_cache(self) -> None:
        self._musa().empty_cache()

    def is_available(self) -> bool:
        try:
            return self._musa().is_available()
        except Exception:
            return False

    def list_devices(self) -> List[str]:
        return [f"musa:{i}" for i in range(self._musa().device_count())]

    def device_name(self, device: str) -> str:
        try:
            return self._musa().get_device_name(self.parse_index(device))
        except Exception:
            return ""

    def get_device_info(self, device: str) -> DeviceInfo:
        index = self.parse_index(device)
        try:
            properties = self._musa().get_device_properties(index)
            return device_info_from_properties(
                index,
                self.device_name(device),
                properties,
            )
        except Exception:
            return DeviceInfo(device_id=index, name=self.device_name(device))

    def status_metadata(self) -> dict[str, object]:
        runtime_version = torch_runtime_version("musa")
        driver_version = status_value_from_label(
            run_status_command(["mthreads-gmi", "-q"]),
            "Driver Version",
        )
        sources = {}
        if runtime_version:
            sources["software.runtime_version"] = "torch.version.musa"
        if driver_version:
            sources["software.driver_version"] = "mthreads-gmi -q"
        return {
            "architecture": "",
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "sources": sources,
        }


    def count_devices_safe(self) -> int:
        import subprocess as _sp
        import sys as _sys
        try:
            result = _sp.run(
                [_sys.executable, "-c",
                 "import torch, torch_musa; print(torch.musa.device_count())"],
                capture_output=True, text=True, timeout=30,
            )
            detected = int(result.stdout.strip()) if result.returncode == 0 else 0
            return self.constrain_visible_device_count(
                detected, "MUSA_VISIBLE_DEVICES"
            )
        except Exception:
            return 0

    @property
    def default_target_hardware(self) -> List[str]:
        return ["MTT-S5000"]
    @property
    def default_entry_point(self) -> str:
        return "main.py::run"
