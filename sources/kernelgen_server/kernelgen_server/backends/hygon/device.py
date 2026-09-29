# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Hygon DCU device implementation."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from functools import lru_cache
from typing import List, Optional, Tuple

import torch

from kernelgen_server.runtime.device import (
    Device,
    DeviceInfo,
    device_info_from_properties,
    run_status_command,
    status_value_from_label,
    torch_runtime_version,
)


_HCU_HARDWARE_ROW = re.compile(
    r"^\s*(\d+)\s+.*?\s+"
    r"((?:[0-9A-Fa-f]{4}:)?[0-9A-Fa-f]{1,2}:[0-9A-Fa-f]{2}\.[0-7])\s*$"
)
_HCU_CARD_KEY = re.compile(r"^card(\d+)$", flags=re.IGNORECASE)
_GENERIC_DEVICE_NAMES = frozenset({"", "device", "dcu", "gpu", "hcu", "unknown"})


def _normalize_bdf(value: str) -> str:
    match = re.fullmatch(
        r"(?:(?P<domain>[0-9A-Fa-f]{4}):)?"
        r"(?P<bus>[0-9A-Fa-f]{1,2}):(?P<device>[0-9A-Fa-f]{2})\."
        r"(?P<function>[0-7])",
        value.strip(),
    )
    if match is None:
        return value.strip().lower()
    return (
        f"{int(match.group('domain') or '0', 16):04x}:"
        f"{int(match.group('bus'), 16):02x}:"
        f"{int(match.group('device'), 16):02x}."
        f"{int(match.group('function'), 16)}"
    )


@lru_cache(maxsize=1)
def _hygon_inventory() -> Tuple[Tuple[int, str, str, str], ...]:
    """Return ``(physical index, BDF, card series, PCI model)`` records."""
    series_by_index = {}
    try:
        output = subprocess.check_output(
            ["hy-smi", "--showproductname", "--json"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        start, end = output.find("{"), output.rfind("}")
        payload = json.loads(output[start : end + 1]) if 0 <= start < end else {}
        for key, value in payload.items():
            match = _HCU_CARD_KEY.match(str(key))
            if match is None or not isinstance(value, dict):
                continue
            series = str(value.get("Card Series", "")).strip()
            if series:
                series_by_index[int(match.group(1))] = series
    except Exception:
        pass

    bdf_by_index = {}
    try:
        output = subprocess.check_output(
            ["hy-smi", "--showhw"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            match = _HCU_HARDWARE_ROW.match(line)
            if match is not None:
                bdf_by_index[int(match.group(1))] = _normalize_bdf(match.group(2))
    except Exception:
        pass

    pci_model_by_bdf = {}
    try:
        output = subprocess.check_output(
            ["lspci", "-Dmm"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            fields = shlex.split(line)
            if len(fields) < 4:
                continue
            model = fields[3].strip()
            if model and not model.lower().startswith("device "):
                pci_model_by_bdf[_normalize_bdf(fields[0])] = model
    except Exception:
        pass

    indexes = sorted(set(series_by_index) | set(bdf_by_index))
    return tuple(
        (
            index,
            bdf_by_index.get(index, ""),
            series_by_index.get(index, ""),
            pci_model_by_bdf.get(bdf_by_index.get(index, ""), ""),
        )
        for index in indexes
    )


def _visible_hcu_selector(logical_index: int) -> Optional[str]:
    for env_var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        value = os.environ.get(env_var)
        if value is None:
            continue
        entries = [item.strip() for item in value.split(",") if item.strip()]
        if logical_index >= len(entries):
            return None
        return entries[logical_index]
    return str(logical_index)


def _detected_hygon_name(logical_index: int, runtime_name: str) -> str:
    selector = _visible_hcu_selector(logical_index)
    if selector is None:
        return runtime_name
    normalized_selector = _normalize_bdf(selector) if ":" in selector else selector
    for physical_index, bdf, series, pci_model in _hygon_inventory():
        if normalized_selector not in {str(physical_index), bdf}:
            continue
        normalized_runtime = runtime_name.strip().lower()
        normalized_series = series.strip().lower()
        if runtime_name and normalized_runtime not in _GENERIC_DEVICE_NAMES:
            if not series or normalized_runtime != normalized_series:
                return runtime_name
        if pci_model:
            return pci_model
        return series or runtime_name
    return runtime_name


class HygonDevice(Device):
    """Hygon HIP backend exposed through PyTorch's ``torch.cuda`` API."""

    backend = "hygon"

    def set_device(self, device: str) -> None:
        torch.cuda.set_device(self.parse_index(device))

    def synchronize(self, device: str) -> None:
        torch.cuda.synchronize(self.parse_index(device))

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def list_devices(self) -> List[str]:
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]

    def device_name(self, device: str) -> str:
        index = self.parse_index(device)
        try:
            runtime_name = str(torch.cuda.get_device_name(index) or "").strip()
        except Exception:
            runtime_name = ""
        return _detected_hygon_name(index, runtime_name)

    def get_device_info(self, device: str) -> DeviceInfo:
        """Return normalized Hygon properties for ``/status`` and profiles."""
        index = self.parse_index(device)
        properties = torch.cuda.get_device_properties(index)
        architecture = str(
            getattr(properties, "gcnArchName", "")
            or getattr(properties, "gcn_arch_name", "")
        )
        return device_info_from_properties(
            index,
            self.device_name(device),
            properties,
            architecture=architecture,
        )

    def status_metadata(self) -> dict[str, object]:
        runtime_version = torch_runtime_version("hip")
        driver_version = status_value_from_label(
            run_status_command(["hy-smi", "--showdriverversion"]),
            "Driver Version",
        )
        sources = {}
        if runtime_version:
            sources["software.runtime_version"] = "torch.version.hip"
        if driver_version:
            sources["software.driver_version"] = "hy-smi --showdriverversion"
        return {
            "architecture": "",
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "sources": sources,
        }

    def count_devices_safe(self) -> int:
        """Detect Hygon HCUs without importing the HIP runtime in the server."""
        import subprocess as _sp

        try:
            output = _sp.check_output(
                ["hy-smi", "--showid"], text=True, timeout=10
            )
            detected = len(
                re.findall(r"^HCU\[\d+\]", output, flags=re.MULTILINE)
            )
            for env_var in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            ):
                visible = self.visible_device_count(env_var)
                if visible is not None:
                    return min(detected, visible)
            return detected
        except Exception:
            return 0
    @property
    def default_target_hardware(self) -> List[str]:
        return [self.device_name("cuda:0")]
