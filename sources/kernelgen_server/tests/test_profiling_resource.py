import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kernelgen_server.profiling.process import (
    ProfileDeadlineExceeded,
    RequestDeadline,
)
from kernelgen_server.profiling.resource import host_global_lock


def test_host_global_lock_wait_uses_request_deadline(tmp_path: Path):
    lock_path = tmp_path / "vendor.lock"
    entered = threading.Event()
    release = threading.Event()

    def hold_lock():
        with host_global_lock(lock_path, RequestDeadline.from_timeout(2)):
            entered.set()
            assert release.wait(timeout=1)

    with ThreadPoolExecutor(max_workers=1) as executor:
        holder = executor.submit(hold_lock)
        assert entered.wait(timeout=1)
        try:
            with pytest.raises(ProfileDeadlineExceeded, match="global profiler lock"):
                with host_global_lock(
                    lock_path,
                    RequestDeadline.from_timeout(0.01),
                ):
                    raise AssertionError("contended lock was acquired")
        finally:
            release.set()
        holder.result(timeout=1)

    assert lock_path.is_file()
