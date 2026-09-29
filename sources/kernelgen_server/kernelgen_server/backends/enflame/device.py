# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Enflame GCU device implementation (ZIXIAO C200 / GCU300)."""

from __future__ import annotations

import ctypes
import os
import re
from functools import lru_cache
from typing import List

import torch

from kernelgen_server.runtime.device import (
    Device,
    DeviceInfo,
    clean_status_value,
    device_info_from_properties,
    run_status_command,
    status_json_from_output,
    status_json_value,
    torch_runtime_version,
)


def _decode_tops_version(value: int) -> str:
    """Decode the integer format documented by TOPS Runtime."""
    if value <= 0:
        return ""
    major = value // 1000
    minor = (value % 1000) // 10
    if major <= 0:
        return ""
    return f"{major}.{minor}"


@lru_cache(maxsize=1)
def _tops_runtime_version() -> str:
    """Query the loaded TOPS Runtime instead of inferring from a package tag."""
    try:
        library = ctypes.CDLL("libtopsrt.so.1")
        query = library.topsRuntimeGetVersion
        query.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int
        value = ctypes.c_int()
        if query(ctypes.byref(value)) != 0:
            return ""
    except (AttributeError, OSError, TypeError, ValueError):
        return ""
    return _decode_tops_version(value.value)


@lru_cache(maxsize=None)
def _efsmi_payload(detail: str) -> object:
    return status_json_from_output(
        run_status_command(["efsmi", "-q", "-d", detail, "--json-format"])
    )


def _visible_gcu_index(logical_index: int) -> int:
    value = os.environ.get("TOPS_VISIBLE_DEVICES")
    if value is None:
        return logical_index
    entries = [item.strip() for item in value.split(",") if item.strip()]
    if logical_index >= len(entries):
        return logical_index
    try:
        return int(entries[logical_index])
    except ValueError:
        return logical_index


def _efsmi_device_record(detail: str, logical_index: int) -> object:
    payload = _efsmi_payload(detail)
    if not isinstance(payload, dict):
        return None
    devices = payload.get("devices", [])
    if not isinstance(devices, list):
        return None
    physical_index = _visible_gcu_index(logical_index)
    return next(
        (
            item
            for item in devices
            if isinstance(item, dict) and item.get("id") == physical_index
        ),
        None,
    )


class EnflameDevice(Device):
    """Enflame backend exposed through the torch_gcu CUDA compatibility layer."""

    backend = "enflame"

    @staticmethod
    def _gcu():
        # The base image ships CPU PyTorch plus torch_gcu's transfer layer.
        # Spawned health/eval workers do not import Triton before set_device(),
        # so load torch_gcu explicitly instead of falling through to torch.cuda.
        import torch_gcu  # noqa: F401

        return torch.gcu

    def set_device(self, device: str) -> None:
        self._gcu().set_device(self.parse_index(device))

    def synchronize(self, device: str) -> None:
        self._gcu().synchronize(self.parse_index(device))

    def empty_cache(self) -> None:
        self._gcu().empty_cache()

    def is_available(self) -> bool:
        return bool(self._gcu().is_available())

    def list_devices(self) -> List[str]:
        return [f"gcu:{index}" for index in range(self._gcu().device_count())]

    def device_name(self, device: str) -> str:
        index = self.parse_index(device)
        observed = status_json_value(
            _efsmi_device_record("DEVICE", index),
            "Dev Name",
            "Device Name",
        )
        if observed:
            return observed
        try:
            return clean_status_value(self._gcu().get_device_name(index))
        except Exception:
            return ""

    def count_devices_safe(self) -> int:
        """Count physical GCUs through efsmi without importing torch_gcu."""
        import subprocess as _sp

        try:
            output = _sp.check_output(
                ["efsmi"],
                text=True,
                stderr=_sp.STDOUT,
                timeout=10,
            )
        except Exception:
            return 0
        detected = len(
            re.findall(r"^\|\s*\d+\s+\S+\s+\|", output, flags=re.MULTILINE)
        )
        return self.constrain_visible_device_count(detected, "TOPS_VISIBLE_DEVICES")

    def get_device_info(self, device: str) -> DeviceInfo:
        index = self.parse_index(device)
        try:
            properties = self._gcu().get_device_properties(index)
            return device_info_from_properties(
                index,
                self.device_name(device),
                properties,
            )
        except Exception:
            return DeviceInfo(device_id=index, name=self.device_name(device))

    def status_metadata(self) -> dict[str, object]:
        runtime_version = _tops_runtime_version()
        runtime_source = "libtopsrt.so.1:topsRuntimeGetVersion"
        if not runtime_version:
            runtime_version = torch_runtime_version("gcu")
            runtime_source = "torch.version.gcu"
        if not runtime_version:
            try:
                runtime = self._gcu()
                for attribute in ("runtime_version", "get_runtime_version"):
                    query = getattr(runtime, attribute, None)
                    if not callable(query):
                        continue
                    try:
                        runtime_version = clean_status_value(query())
                    except Exception:
                        continue
                    if runtime_version:
                        runtime_source = f"torch.gcu.{attribute}()"
                        break
            except Exception:
                pass
        driver_version = status_json_value(
            _efsmi_device_record("DRIVER", 0),
            "Ver",
            "Driver Version",
        )
        sources = {}
        if runtime_version:
            sources["software.runtime_version"] = runtime_source
        if driver_version:
            sources["software.driver_version"] = "efsmi -q -d DRIVER --json-format"
        return {
            "architecture": "",
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "sources": sources,
        }

    @property
    def default_target_hardware(self) -> List[str]:
        return [self.device_name("gcu:0")]
    @property
    def default_entry_point(self) -> str:
        return "main.py::run"
