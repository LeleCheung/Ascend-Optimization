# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""KunlunXin (百度昆仑芯) XPU device implementation."""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from typing import List, Optional, Tuple

import torch

from kernelgen_server.runtime.device import Device, run_status_command, status_xml_value


_XPU_LIST_LINE = re.compile(r"^XPU\s+(\d+):\s+(.+?)\s*$")


@lru_cache(maxsize=1)
def _xpu_smi_inventory() -> Tuple[Tuple[int, str, str], ...]:
    """Return ``(physical index, product name, UUID)`` records from xpu-smi."""
    try:
        output = subprocess.check_output(
            ["xpu-smi", "-L"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ()

    records = []
    for line in output.splitlines():
        match = _XPU_LIST_LINE.match(line)
        if match is None:
            continue
        descriptor = match.group(2).strip()
        uuid = ""
        if " (UUID:" in descriptor and descriptor.endswith(")"):
            descriptor, uuid = descriptor.rsplit(" (UUID:", 1)
            uuid = uuid[:-1].strip()
        name = descriptor.strip()
        if name:
            records.append((int(match.group(1)), name, uuid))
    return tuple(records)


def _visible_xpu_selector(logical_index: int) -> Optional[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return str(logical_index)
    entries = [item.strip() for item in value.split(",") if item.strip()]
    if logical_index >= len(entries):
        return None
    return entries[logical_index]


class KunlunxinDevice(Device):
    """KunlunXin XPU backend — uses CUDA-compatible torch.cuda API.

    - Device API: torch.cuda (XPU exposed via CUDA-compat layer)
    - Query cmd: xpu-smi
    - Capabilities: fp64=False, bf16=True, int64=True
    - Triton extra: xpu
    - Default timing: Triton ``do_bench``
    """

    backend = "kunlunxin"

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
        index = self.parse_index(device)
        selector = _visible_xpu_selector(index)
        if selector is not None:
            for physical_index, product_name, uuid in _xpu_smi_inventory():
                if selector == str(physical_index) or (
                    uuid and (selector == uuid or uuid.startswith(selector))
                ):
                    return product_name
        try:
            return torch.cuda.get_device_name(index)
        except Exception:
            return ""

    def status_metadata(self) -> dict[str, object]:
        output = run_status_command(["xpu-smi", "-q", "-x"])
        architecture = status_xml_value(output, "product_architecture")
        runtime_version = status_xml_value(output, "cuda_version")
        driver_version = status_xml_value(output, "driver_version")
        sources = {}
        if architecture:
            sources["target.architecture"] = "xpu-smi -q -x"
        if runtime_version:
            sources["software.runtime_version"] = "xpu-smi -q -x"
        if driver_version:
            sources["software.driver_version"] = "xpu-smi -q -x"
        return {
            "architecture": architecture,
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "sources": sources,
        }

    def count_devices_safe(self) -> int:
        """KunlunXin: use xpu-smi to detect. Returns 0 if not a KunlunXin machine."""
        import subprocess as _sp
        try:
            out = _sp.check_output(["xpu-smi", "-L"], text=True, timeout=10)
            detected = len(
                re.findall(r"^XPU\s+\d+:", out, flags=re.MULTILINE)
            )
            return self.constrain_visible_device_count(
                detected, "CUDA_VISIBLE_DEVICES"
            )
        except Exception:
            return 0

    @property
    def default_target_hardware(self) -> List[str]:
        return [self.device_name("cuda:0")]
    @property
    def default_entry_point(self) -> str:
        return "main.py::run"
