from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from kernelgen_server.runtime.device_pool import (
    DevicePool,
    DeviceSlot,
    probe_startup_slots,
)


def test_startup_probe_preserves_device_order_and_failures():
    barrier = threading.Barrier(3)

    def probe(device: str) -> None:
        barrier.wait(timeout=1)
        if device == "cuda:1":
            raise RuntimeError("device unavailable")

    slots = probe_startup_slots(["cuda:0", "cuda:1", "cuda:2"], probe)

    assert [slot.device for slot in slots] == ["cuda:0", "cuda:1", "cuda:2"]
    assert [slot.state for slot in slots] == [
        "available",
        "broken",
        "available",
    ]
    assert slots[1].reason == "RuntimeError: device unavailable"
    assert slots[1].last_incident == slots[1].reason
    assert slots[1].incidents == 1


def test_acquire_rotates_available_devices():
    async def exercise(pool: DevicePool) -> None:
        first = await pool.acquire()
        await pool.release(first)
        second = await pool.acquire()
        await pool.release(second)

        assert first == "cuda:0"
        assert second == "cuda:1"
        assert pool.snapshot(probe_timeout_seconds=30)["available"] == 2

    with ThreadPoolExecutor(max_workers=2) as executor:
        pool = DevicePool(
            [
                DeviceSlot("cuda:0", "available"),
                DeviceSlot("cuda:1", "available"),
            ],
            worker_count=2,
            executor=executor,
            probe=lambda device: None,
        )
        asyncio.run(exercise(pool))


def test_recovery_probe_updates_incident_and_restores_the_slot():
    async def exercise(pool: DevicePool) -> None:
        device = await pool.acquire()
        broken_reason = await pool.probe_after_failure(device, "timeout")
        checking = pool.snapshot(probe_timeout_seconds=30)
        assert checking["checking"] == 1
        assert checking["incidents"] == 1
        assert checking["recovered"] == 1

        await pool.release(device, broken_reason=broken_reason)
        restored = pool.snapshot(probe_timeout_seconds=30)
        assert restored["healthy"] == 1
        assert restored["checking"] == 0
        assert restored["broken"] == 0
        assert restored["available"] == 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        pool = DevicePool(
            [DeviceSlot("cuda:0", "available")],
            worker_count=1,
            executor=executor,
            probe=lambda device: None,
        )
        asyncio.run(exercise(pool))


def test_failed_recovery_probe_quarantines_the_slot():
    def failed_probe(device: str) -> None:
        raise RuntimeError("queue unhealthy")

    async def exercise(pool: DevicePool) -> None:
        device = await pool.acquire()
        broken_reason = await pool.probe_after_failure(device, "timeout")
        assert "recovery probe failed" in broken_reason
        await pool.release(device, broken_reason=broken_reason)

        status = pool.snapshot(probe_timeout_seconds=30)
        assert status["healthy"] == 0
        assert status["broken"] == 1
        assert status["available"] == 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        pool = DevicePool(
            [DeviceSlot("cuda:0", "available")],
            worker_count=1,
            executor=executor,
            probe=failed_probe,
        )
        asyncio.run(exercise(pool))


def test_repeated_failures_remain_available_when_probe_passes():
    async def exercise(pool: DevicePool) -> None:
        for _ in range(5):
            device = await pool.acquire()
            broken_reason = await pool.probe_after_failure(device, "timeout")
            assert broken_reason == ""
            await pool.release(device, broken_reason=broken_reason)

        status = pool.snapshot(probe_timeout_seconds=30)
        assert status["incidents"] == 5
        assert status["recovered"] == 5
        assert status["broken"] == 0
        assert status["available"] == 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        pool = DevicePool(
            [DeviceSlot("cuda:0", "available")],
            worker_count=1,
            executor=executor,
            probe=lambda device: None,
        )
        asyncio.run(exercise(pool))
