import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kernelgen_server.runtime.isolated import (
    _enter_isolated_process_group,
    _stop_process,
)


def _spawn_stubborn_descendant(connection, ready_path: str) -> None:
    _enter_isolated_process_group()
    script = (
        "import os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", script, ready_path])
    deadline = time.monotonic() + 5
    while not Path(ready_path).is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("descendant did not become ready")
        time.sleep(0.01)
    connection.send((os.getpid(), os.getpgrp(), child.pid))
    connection.close()
    while True:
        time.sleep(1)


def _pid_is_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    return len(fields) >= 3 and fields[2] not in {"X", "Z"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_stop_process_terminates_entire_isolated_process_group(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_spawn_stubborn_descendant,
        args=(child, str(tmp_path / "descendant.ready")),
        daemon=False,
    )
    descendant_pid = None
    process.start()
    child.close()
    try:
        assert parent.poll(10)
        worker_pid, process_group, descendant_pid = parent.recv()
        assert worker_pid == process.pid
        assert process_group == process.pid
        assert _pid_is_running(descendant_pid)

        _stop_process(process, grace_seconds=0.2)

        assert not process.is_alive()
        deadline = time.monotonic() + 3
        while _pid_is_running(descendant_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_is_running(descendant_pid)
    finally:
        parent.close()
        if process.is_alive():
            _stop_process(process, grace_seconds=0.2)
        if descendant_pid is not None and _pid_is_running(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)
