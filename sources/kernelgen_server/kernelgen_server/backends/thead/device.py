# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""T-Head (平头哥) PPU device implementation (PPU-ZW810E)."""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Callable, List, Optional

import torch

from kernelgen_server.runtime.device import (
    Device,
    run_status_command,
    status_value_from_label,
)


class TheadDevice(Device):
    """T-Head PPU backend — CUDA-compatible API.

    - Device API: torch.cuda (PPU uses CUDA-compatible API)
    - Query cmd: ppu-smi
    - Capabilities: fp64=True, bf16=True, int64=True
    - Default timing: Triton ``do_bench``
    """

    backend = "thead"
    PPU_SDK_ROOT = "/usr/local/PPU_SDK"
    CUDA_SDK_ROOT = f"{PPU_SDK_ROOT}/CUDA_SDK"

    @classmethod
    def _configure_sdk_environment(cls) -> None:
        """Expose the PPU toolchain to the vendor Triton compiler."""
        if os.path.isdir(cls.PPU_SDK_ROOT):
            os.environ.setdefault("PPU_SDK", cls.PPU_SDK_ROOT)
        if os.path.isdir(cls.CUDA_SDK_ROOT):
            os.environ.setdefault("CUDA_PATH", cls.CUDA_SDK_ROOT)
            os.environ.setdefault("CUDA_HOME", cls.CUDA_SDK_ROOT)

    def set_device(self, device: str) -> None:
        torch.cuda.set_device(self.parse_index(device))

    def synchronize(self, device: str) -> None:
        torch.cuda.synchronize(self.parse_index(device))

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def list_devices(self) -> List[str]:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    def device_name(self, device: str) -> str:
        try:
            return torch.cuda.get_device_name(self.parse_index(device))
        except Exception:
            return ""

    @classmethod
    def _ppu_smi(cls) -> str:
        return shutil.which("ppu-smi") or f"{cls.PPU_SDK_ROOT}/ppu-smi/bin/ppu-smi"

    def status_metadata(self) -> dict[str, object]:
        command = self._ppu_smi()
        device_output = run_status_command([command, "-q"])
        version_output = run_status_command([command, "-q", "-d", "VERSION"])
        architecture = status_value_from_label(
            device_output,
            "Product Architecture",
            "Architecture",
        )
        runtime_version = status_value_from_label(version_output, "SDK Version")
        driver_version = status_value_from_label(
            "\n".join((version_output, device_output)),
            "Driver Version",
        )
        sources = {}
        if architecture:
            sources["target.architecture"] = "ppu-smi -q"
        if runtime_version:
            sources["software.runtime_version"] = "ppu-smi -q -d VERSION"
        if driver_version:
            sources["software.driver_version"] = "ppu-smi -q -d VERSION"
        return {
            "architecture": architecture,
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "sources": sources,
        }


    def count_devices_safe(self) -> int:
        """T-Head PPU: use ppu-smi to detect. Returns 0 if not a PPU machine."""
        import subprocess as _sp

        try:
            commands = (
                shutil.which("ppu-smi"),
                "/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi",
            )
            for command in commands:
                if not command:
                    continue
                try:
                    out = _sp.check_output(
                        [command, "-L"], text=True, timeout=10
                    )
                except FileNotFoundError:
                    continue
                detected = len(
                    re.findall(r"^PPU\s+\d+:", out, flags=re.MULTILINE)
                )
                if detected:
                    self._configure_sdk_environment()
                return self.constrain_visible_device_count(
                    detected, "CUDA_VISIBLE_DEVICES"
                )
            return 0
        except Exception:
            return 0

    @property
    def default_target_hardware(self) -> List[str]:
        return ["PPU-ZW810E"]
    @property
    def default_entry_point(self) -> str:
        return "main.py::run"
