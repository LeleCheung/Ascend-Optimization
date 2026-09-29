"""Exclusive device-slot scheduling and health-state management."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


DeviceProbe = Callable[[str], None]


class NoHealthyDeviceError(RuntimeError):
    """No device slot remains eligible for scheduling."""


def _exception_summary(exc: Exception) -> str:
    detail = str(exc).strip().splitlines()
    message = detail[-1] if detail else ""
    return f"{type(exc).__name__}: {message}"


@dataclass
class DeviceSlot:
    """Mutable health and scheduling state for one visible device."""

    device: str
    state: str
    reason: str = ""
    last_incident: str = ""
    incidents: int = 0
    recovered: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "state": self.state,
            "reason": self.reason,
            "last_incident": self.last_incident,
            "incidents": self.incidents,
            "recovered": self.recovered,
        }


def probe_startup_slots(
    devices: list[str],
    probe: DeviceProbe,
) -> list[DeviceSlot]:
    """Probe all visible devices concurrently while preserving device order."""

    def check(device: str) -> DeviceSlot:
        try:
            probe(device)
        except Exception as exc:
            error = _exception_summary(exc)
            return DeviceSlot(
                device=device,
                state="broken",
                reason=error,
                last_incident=error,
                incidents=1,
            )
        return DeviceSlot(device=device, state="available")

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        return list(executor.map(check, devices))


class DevicePool:
    """Own exclusive device tokens, queueing, and recovery-probe state."""

    def __init__(
        self,
        slots: list[DeviceSlot],
        *,
        worker_count: int,
        executor: Executor,
        probe: DeviceProbe,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if not slots:
            raise ValueError("at least one device slot is required")
        devices = [slot.device for slot in slots]
        if len(devices) != len(set(devices)):
            raise ValueError("device slots must be unique")

        self.worker_count = worker_count
        self._devices = devices
        self._slots = {slot.device: slot for slot in slots}
        self._available = deque(
            slot.device for slot in slots if slot.state == "available"
        )
        self._condition = asyncio.Condition()
        self._worker_slots = asyncio.Semaphore(worker_count)
        self._executor = executor
        self._probe = probe
        self._active = 0
        self._waiting = 0

    @property
    def healthy_devices(self) -> list[str]:
        return [
            device
            for device in self._devices
            if self._slots[device].state != "broken"
        ]

    async def acquire(self) -> str:
        self._waiting += 1
        worker_acquired = False
        try:
            await self._worker_slots.acquire()
            worker_acquired = True
            async with self._condition:
                while True:
                    selected = None
                    for _ in range(len(self._available)):
                        candidate = self._available.popleft()
                        if selected is None:
                            selected = candidate
                        else:
                            self._available.append(candidate)
                    if selected is not None:
                        slot = self._slots[selected]
                        slot.state = "active"
                        slot.reason = ""
                        self._active += 1
                        return selected

                    if not any(
                        slot.state != "broken" for slot in self._slots.values()
                    ):
                        raise NoHealthyDeviceError(
                            "all device slots are broken; restart the affected "
                            "device runtime and the server"
                        )
                    await self._condition.wait()
        except BaseException:
            if worker_acquired:
                self._worker_slots.release()
            raise
        finally:
            self._waiting -= 1

    async def release(
        self,
        device: str,
        *,
        broken_reason: str = "",
    ) -> None:
        try:
            async with self._condition:
                self._active -= 1
                slot = self._slots[device]
                if broken_reason:
                    slot.state = "broken"
                    slot.reason = broken_reason
                else:
                    slot.state = "available"
                    slot.reason = ""
                    self._available.append(device)
                self._condition.notify_all()
        finally:
            self._worker_slots.release()

    async def probe_after_failure(
        self,
        device: str,
        cause: str,
    ) -> str:
        async with self._condition:
            slot = self._slots[device]
            slot.state = "checking"
            slot.reason = cause
            slot.last_incident = cause
            slot.incidents += 1

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._probe, device)
        except Exception as exc:
            return (
                f"{cause}; recovery probe failed: "
                f"{_exception_summary(exc)}"
            )

        async with self._condition:
            slot = self._slots[device]
            slot.recovered += 1
        return ""

    def snapshot(self, *, probe_timeout_seconds: float) -> dict[str, Any]:
        broken_count = sum(
            slot.state == "broken" for slot in self._slots.values()
        )
        checking_count = sum(
            slot.state == "checking" for slot in self._slots.values()
        )
        healthy_count = len(self._devices) - broken_count - checking_count
        return {
            "device_slots": len(self._devices),
            "healthy": healthy_count,
            "checking": checking_count,
            "broken": broken_count,
            "incidents": sum(
                slot.incidents for slot in self._slots.values()
            ),
            "recovered": sum(
                slot.recovered for slot in self._slots.values()
            ),
            "probe_timeout_seconds": probe_timeout_seconds,
            "max_active": min(self.worker_count, healthy_count),
            "active": self._active,
            "waiting": self._waiting,
            "available": len(self._available),
            "slots": [
                self._slots[device].as_dict() for device in self._devices
            ],
        }
