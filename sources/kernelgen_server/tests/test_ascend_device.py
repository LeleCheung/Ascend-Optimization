import pytest

pytest.importorskip("torch")

from kernelgen_server.backends.ascend.device import NpuDevice
from kernelgen_server.runtime import device as device_runtime


def test_npu_event_timing_is_not_supported():
    with pytest.raises(ValueError, match="perf_mode must be"):
        NpuDevice(perf_mode="npu-event")

    with pytest.raises(ValueError, match="Unknown timing strategy"):
        device_runtime.configure_device(
            "npu",
            timing_strategy="npu-event",
            perf_mode="npu-event",
        )
