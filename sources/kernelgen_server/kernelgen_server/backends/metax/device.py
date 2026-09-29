# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""MetaX (沐曦) CUDA-compatible device implementation (C550)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from kernelgen_server.runtime.device import (
    Device,
    run_status_command,
    status_json_from_output,
    status_json_value,
    torch_runtime_version,
)


class MetaxDevice(Device):
    """MetaX (沐曦) backend — CUDA-compatible API.

    - Device API: torch.cuda (shares CUDA namespace)
    - Query cmd: mx-smi
    - Capabilities: fp64=True, bf16=True, int64=True
    - Default timing: Triton ``do_bench``
    """

    backend = "metax"

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

    def status_metadata(self) -> dict[str, object]:
        output = run_status_command(["mx-smi", "--show-version", "-j"])
        payload = status_json_from_output(output)
        runtime_version = status_json_value(payload, "maca_version")
        runtime_source = "mx-smi --show-version -j"
        if not runtime_version:
            runtime_version = torch_runtime_version("maca")
            runtime_source = "torch.version.maca"
        driver_version = status_json_value(payload, "driver_version")
        sources = {}
        if runtime_version:
            sources["software.runtime_version"] = runtime_source
        if driver_version:
            sources["software.driver_version"] = "mx-smi --show-version -j"
        return {
            "architecture": "",
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "sources": sources,
        }


    def count_devices_safe(self) -> int:
        """MetaX: use mx-smi to detect. Returns 0 if not a MetaX machine."""
        import subprocess as _sp
        try:
            out = _sp.check_output(["mx-smi", "-L"], text=True, timeout=10)
            detected = len(
                [line for line in out.strip().splitlines() if "GPU" in line or "Device" in line]
            )
            return self.constrain_visible_device_count(
                detected, "CUDA_VISIBLE_DEVICES"
            )
        except Exception:
            return 0

    @property
    def default_target_hardware(self) -> List[str]:
        return ["MetaX-C550"]
    @property
    def default_entry_point(self) -> str:
        return "main.py::run"
