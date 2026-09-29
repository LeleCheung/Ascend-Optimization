# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA CUDA device implementation."""

from __future__ import annotations

from typing import List

import torch

from kernelgen_server.runtime.device import (
    Device,
    DeviceInfo,
    clean_status_value,
    run_status_command,
    torch_runtime_version,
)


class CudaDevice(Device):
    """NVIDIA CUDA backend. Timing defaults to Triton's ``do_bench``."""

    backend = "cuda"

    def set_device(self, device: str) -> None:
        torch.cuda.set_device(self.parse_index(device))

    def synchronize(self, device: str) -> None:
        torch.cuda.synchronize(device)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def list_devices(self) -> List[str]:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    def device_name(self, device: str) -> str:
        return torch.cuda.get_device_name(torch.device(device).index)

    def count_devices_safe(self) -> int:
        """CUDA: prefer CUDA_VISIBLE_DEVICES > nvidia-smi > subprocess torch."""
        import subprocess as _sp

        visible = self.visible_device_count("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            return visible
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True, timeout=10
            )
            return len(out.strip().splitlines())
        except Exception:
            pass
        # Fallback to base (subprocess torch.cuda.device_count)
        return super().count_devices_safe()

    @property
    def default_target_hardware(self) -> List[str]:
        return ["H100"]

    @property
    def default_entry_point(self) -> str:
        return "main.py::kernel_function"

    def get_device_info(self, device: str) -> DeviceInfo:
        idx = self.parse_index(device)
        try:
            props = torch.cuda.get_device_properties(idx)
            l2 = getattr(props, "L2_cache_size", 0) or 0
            sm = getattr(props, "multi_processor_count", 0) or 0
            cap = f"{props.major}.{props.minor}" if hasattr(props, "major") else ""
            mem = getattr(props, "total_memory", 0) or 0
            return DeviceInfo(
                device_id=idx,
                name=props.name,
                l2_cache_size=l2,
                num_sm=sm,
                compute_capability=cap,
                total_memory=mem,
            )
        except Exception:
            return DeviceInfo(device_id=idx, name=self.device_name(device))
    def status_metadata(self) -> dict[str, object]:
        runtime_version = torch_runtime_version("cuda")
        driver_output = run_status_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        driver_version = clean_status_value(
            next(iter(driver_output.splitlines()), "")
        )
        sources = {}
        if runtime_version:
            sources["software.runtime_version"] = "torch.version.cuda"
        if driver_version:
            sources["software.driver_version"] = "nvidia-smi"
        return {
            "architecture": "",
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "sources": sources,
        }
