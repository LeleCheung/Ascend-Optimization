# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Tsingmicro TXDA device implementation (TX8110)."""

from __future__ import annotations

from typing import List

import torch

from kernelgen_server.runtime.device import Device, DeviceInfo


class TxdaDevice(Device):
    """Tsingmicro TXDA backend.

    - Device API: torch.txda (via the torch_txda package, PrivateUse1)
    - Query cmd: tsm_smi
    - Default timing: Triton ``do_bench`` (the tsingmicro Triton fork ships
      ``triton.testing``); ``TXDA_LAUNCH_KERNEL_SYNC=1`` keeps launches
      synchronous, which do_bench tolerates.
    """

    backend = "txda"

    def _txda(self):
        # Lazy import: torch_txda initializes the HPGR device manager at import
        # time, so keep it out of module load.
        import torch_txda  # noqa: F401

        return torch.txda

    def set_device(self, device: str) -> None:
        self._txda().set_device(self.parse_index(device))

    def synchronize(self, device: str) -> None:
        self._txda().synchronize()

    def empty_cache(self) -> None:
        self._txda().empty_cache()

    def is_available(self) -> bool:
        try:
            return self._txda().is_available()
        except Exception:
            return False

    def list_devices(self) -> List[str]:
        return [f"txda:{i}" for i in range(self._txda().device_count())]

    def device_name(self, device: str) -> str:
        try:
            return self._txda().get_device_name(self.parse_index(device))
        except Exception:
            return "Tsingmicro TX8110"

    def count_devices_safe(self) -> int:
        import subprocess as _sp
        import sys as _sys

        try:
            result = _sp.run(
                [
                    _sys.executable,
                    "-c",
                    "import torch, torch_txda; print(torch.txda.device_count())",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            detected = int(result.stdout.strip()) if result.returncode == 0 else 0
            return self.constrain_visible_device_count(
                detected, "TXDA_VISIBLE_DEVICES"
            )
        except Exception:
            return 0

    @property
    def default_target_hardware(self) -> List[str]:
        return ["TX8110"]

    @property
    def default_entry_point(self) -> str:
        return "main.py::run"

    def get_device_info(self, device: str) -> DeviceInfo:
        idx = self.parse_index(device)
        try:
            props = self._txda().get_device_properties(idx)
            return DeviceInfo(
                device_id=idx,
                name=getattr(props, "name", "") or self.device_name(device),
                l2_cache_size=int(getattr(props, "L2_cache_size", 0) or 0),
                num_sm=int(getattr(props, "multi_processor_count", 0) or 0),
                compute_capability=str(getattr(props, "gcnArchName", "") or ""),
                total_memory=int(getattr(props, "total_memory", 0) or 0),
            )
        except Exception:
            return DeviceInfo(device_id=idx, name=self.device_name(device))
