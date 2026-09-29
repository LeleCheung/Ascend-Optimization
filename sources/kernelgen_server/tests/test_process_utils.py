import os
from pathlib import Path

import pytest

from kernelgen_server.process_utils import process_start_identity, process_is_alive


def test_identity_matches_current_process_but_not_reused_pid():
    identity = process_start_identity(os.getpid())
    assert identity and process_is_alive(os.getpid(), identity)
    assert not process_is_alive(os.getpid(), identity + "0")
    assert process_start_identity(0) is None


@pytest.mark.parametrize("state", ["Z", "X"])
def test_dead_process_is_not_a_live_owner(monkeypatch, state):
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: "12 (name with ) spaces) " + " ".join([state] + ["0"] * 18 + ["123"]))
    assert process_start_identity(12) is None


def test_unreadable_identity_is_an_error_not_pid_fallback(monkeypatch):
    def denied(*a, **k):
        raise PermissionError("unreadable")
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(RuntimeError, match="cannot determine"):
        process_start_identity(12)


def test_management_bootstraps_without_installed_dependencies():
    import subprocess
    import sys
    from kernelgen_server import management
    result = subprocess.run([sys.executable, "-S", management.__file__], text=True, capture_output=True)
    assert "usage: management.py" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
