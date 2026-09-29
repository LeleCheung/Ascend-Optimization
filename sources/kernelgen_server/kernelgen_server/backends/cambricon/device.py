# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Cambricon (寒武纪) MLU device implementation (MLU590-M9DE)."""

from __future__ import annotations

import ctypes
import re
from functools import lru_cache
from typing import Any, Callable, List, Optional

import torch

from kernelgen_server.runtime.device import (
    Device,
    DeviceInfo,
    clean_status_value,
    device_info_from_properties,
    run_status_command,
    status_value_from_label,
    torch_runtime_version,
)


@lru_cache(maxsize=1)
def _cndev_driver_version() -> str:
    """Read the loaded MLU driver version through the vendor CNDEV API."""
    initialized = False
    library = None
    try:
        library = ctypes.CDLL("libcndev.so")
        initialize = library.cndevInit
        initialize.argtypes = [ctypes.c_int32]
        initialize.restype = ctypes.c_int
        if initialize(0) != 0:
            return ""
        initialized = True

        query = library.cndevGetDriverVersion
        query.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        query.restype = ctypes.c_int
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        build = ctypes.c_uint32()
        if query(
            ctypes.byref(major),
            ctypes.byref(minor),
            ctypes.byref(build),
        ) != 0:
            return ""
        return f"{major.value}.{minor.value}.{build.value}"
    except (AttributeError, OSError, TypeError, ValueError):
        return ""
    finally:
        if initialized and library is not None:
            try:
                release = library.cndevRelease
                release.argtypes = []
                release.restype = ctypes.c_int
                release()
            except (AttributeError, OSError, TypeError, ValueError):
                pass


def _cnmon_driver_version() -> str:
    value = status_value_from_label(
        run_status_command(["cnmon", "info", "-d"]),
        "Driver",
    )
    if re.fullmatch(r"v\d+(?:\.\d+)+", value, flags=re.IGNORECASE):
        return value[1:]
    return value


class CambriconDevice(Device):
    """Cambricon MLU backend.

    - Device API: torch.mlu (via torch_mlu / cambricon-pytorch)
    - Query cmd: cnmon
    - Capabilities: fp64=False, bf16=True, int64=True
    - Dispatch key: PrivateUse1
    - Default timing: wall-time
    """

    backend = "mlu"

    def _mlu(self):
        import torch_mlu  # noqa: F401
        return torch.mlu

    def set_device(self, device: str) -> None:
        self._mlu().set_device(self.parse_index(device))

    def synchronize(self, device: str) -> None:
        self._mlu().synchronize()

    def empty_cache(self) -> None:
        self._mlu().empty_cache()

    def is_available(self) -> bool:
        try:
            return self._mlu().is_available()
        except Exception:
            return False

    def list_devices(self) -> List[str]:
        return [f"mlu:{i}" for i in range(self._mlu().device_count())]

    def device_name(self, device: str) -> str:
        try:
            return self._mlu().get_device_name(self.parse_index(device))
        except Exception:
            return ""

    def get_device_info(self, device: str) -> DeviceInfo:
        index = self.parse_index(device)
        try:
            properties = self._mlu().get_device_properties(index)
            architecture = clean_status_value(
                getattr(properties, "isa_version", "")
            )
            return device_info_from_properties(
                index,
                self.device_name(device),
                properties,
                architecture=architecture,
            )
        except Exception:
            return DeviceInfo(device_id=index, name=self.device_name(device))

    def status_metadata(self) -> dict[str, object]:
        runtime_version = torch_runtime_version("mlu")
        driver_version = _cndev_driver_version()
        driver_source = "libcndev.so:cndevGetDriverVersion"
        try:
            runtime = self._mlu()
            properties = runtime.get_device_properties(0)
            if not driver_version:
                driver_version = clean_status_value(
                    getattr(properties, "driver_version", "")
                )
                if driver_version:
                    driver_source = "torch.mlu device properties"
            if not driver_version:
                for attribute in ("driver_version", "get_driver_version"):
                    query = getattr(runtime, attribute, None)
                    if not callable(query):
                        continue
                    try:
                        driver_version = clean_status_value(query())
                    except Exception:
                        continue
                    if driver_version:
                        driver_source = f"torch.mlu.{attribute}()"
                        break
        except Exception:
            pass
        if not driver_version:
            driver_version = _cnmon_driver_version()
            if driver_version:
                driver_source = "cnmon info -d"
        sources = {}
        if runtime_version:
            sources["software.runtime_version"] = "torch.version.mlu"
        if driver_version:
            sources["software.driver_version"] = driver_source
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
                 "import torch, torch_mlu; print(torch.mlu.device_count())"],
                capture_output=True, text=True, timeout=30,
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception:
            return 0

    @property
    def default_target_hardware(self) -> List[str]:
        return ["MLU590-M9DE"]
    @property
    def default_entry_point(self) -> str:
        return "main.py::run"
