import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kernelgen_server.runtime.process_control import terminate_process_tree


def _pid_is_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    return len(fields) >= 3 and fields[2] not in {"X", "Z"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX sessions are required")
def test_terminate_process_tree_catches_descendant_in_separate_group(tmp_path: Path):
    child_pid_path = tmp_path / "child.pid"
    child_script = (
        "import os, pathlib, signal, sys, time\n"
        "os.setpgrp()\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    leader_script = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
        "while True: time.sleep(1)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader_script, child_script, str(child_pid_path)],
        start_new_session=True,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert os.getsid(child_pid) == process.pid
        assert os.getpgid(child_pid) != process.pid

        result = terminate_process_tree(process, grace_seconds=0.2)

        assert result.group_signaled is True
        assert result.forced is True
        assert process.poll() is not None
        deadline = time.monotonic() + 3
        while _pid_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_is_running(child_pid)
    finally:
        if process.poll() is None:
            terminate_process_tree(process, grace_seconds=0.1)
        if child_pid is not None and _pid_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)
