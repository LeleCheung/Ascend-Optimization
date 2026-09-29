"""GPU device pool with lock-file based allocation."""

import errno
import logging
import os
import re
import subprocess
import time

from .platform import PlatformInfo, PLATFORM_REGISTRY, detect_platform

logger = logging.getLogger(__name__)


class GPUPool:
    def __init__(
        self,
        device_ids: list[int] | None = None,
        vendor: str | None = None,
        platform: PlatformInfo | None = None,
        mode: str = "exclusive",
        lock_dir: str = "/tmp/kgrunner_device_locks",
    ):
        if platform is not None:
            self.platform = platform
        else:
            self.platform = detect_platform(vendor)
        self.mode = mode
        self.lock_dir = lock_dir
        os.makedirs(lock_dir, exist_ok=True)

        if device_ids is not None:
            self.device_ids = device_ids
        else:
            self.device_ids = self._detect_devices()

        logger.info("GPUPool: %s devices %s", self.platform.vendor_name, self.device_ids)

    def _detect_devices(self) -> list[int]:
        if not self.platform.device_query_cmd:
            return [0]
        try:
            cmd_parts = self.platform.device_query_cmd.split()
            result = subprocess.run(
                cmd_parts, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return self._parse_device_ids(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        logger.warning("Failed to detect devices, defaulting to [0]")
        return [0]

    def _parse_device_ids(self, output: str) -> list[int]:
        ids = []
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.isdigit():
                ids.append(int(line))
                continue
            m = re.match(r".*(?:ID|id)\s*[:：]\s*(\d+)", line)
            if m:
                ids.append(int(m.group(1)))
                continue
            m = re.match(r"(?:Card|GPU|Device|NPU)\s+(\d+)", line)
            if m:
                ids.append(int(m.group(1)))
                continue
        seen = set()
        unique = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                unique.append(i)
        return unique if unique else [0]

    def _lock_path(self, device_id: int) -> str:
        return os.path.join(self.lock_dir, f"device_{device_id}.lock")

    def acquire(self, timeout: float | None = None) -> int:
        """Acquire a free device. Blocks until one is available or timeout."""
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            device_id = self._try_acquire()
            if device_id is not None:
                return device_id

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No GPU available within {timeout}s "
                    f"(pool: {self.device_ids})"
                )
            time.sleep(0.5)

    def _try_acquire(self) -> int | None:
        for device_id in self.device_ids:
            lock_path = self._lock_path(device_id)
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n{time.time()}\n".encode())
                os.close(fd)
                return device_id
            except OSError as e:
                if e.errno != errno.EEXIST:
                    continue

            if self._is_lock_stale(lock_path):
                try:
                    os.remove(lock_path)
                except OSError:
                    continue
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, f"{os.getpid()}\n{time.time()}\n".encode())
                    os.close(fd)
                    return device_id
                except OSError:
                    continue
        return None

    def release(self, device_id: int) -> None:
        lock_path = self._lock_path(device_id)
        if os.path.exists(lock_path):
            try:
                with open(lock_path) as f:
                    pid = int(f.read().strip().split("\n")[0])
                if pid == os.getpid():
                    os.remove(lock_path)
            except (OSError, ValueError, IndexError):
                pass

    def _is_lock_stale(self, lock_path: str) -> bool:
        try:
            with open(lock_path) as f:
                pid = int(f.read().strip().split("\n")[0])
            os.kill(pid, 0)
            return False
        except PermissionError:
            return False
        except (OSError, ValueError, IndexError):
            return True

    def release_all(self) -> None:
        for device_id in self.device_ids:
            lock_path = self._lock_path(device_id)
            if os.path.exists(lock_path):
                try:
                    with open(lock_path) as f:
                        pid = int(f.read().strip().split("\n")[0])
                    if pid == os.getpid() or self._is_lock_stale(lock_path):
                        os.remove(lock_path)
                except (OSError, ValueError, IndexError):
                    pass

    def available_count(self) -> int:
        count = 0
        for device_id in self.device_ids:
            lock_path = self._lock_path(device_id)
            if not os.path.exists(lock_path) or self._is_lock_stale(lock_path):
                count += 1
        return count

    @property
    def size(self) -> int:
        return len(self.device_ids)

    def __enter__(self) -> "GPUPool":
        return self

    def __exit__(self, *exc) -> None:
        self.release_all()
