"""Regression tests for remote KGS validation after editable installation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kernelgen_server import management as remote_server


def test_bootstrap_source_runs_before_client_install(tmp_path):
    script = """
import os
import runpy
import sys
management = runpy.run_path(sys.argv[1])
assert management['_process_start'](os.getpid()) is not None
assert 'kernelgen_client' not in sys.modules
"""
    source = Path(__file__).resolve().parents[1] / "kernelgen_server/management.py"
    subprocess.run(
        [sys.executable, "-S", "-c", script, str(source)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def test_remote_validation_uses_fresh_python_process(tmp_path, monkeypatch):
    kgs_root = tmp_path / "kgs"
    module_path = kgs_root / "kernelgen_server" / "__init__.py"
    commands: list[list[str]] = []

    def fail_in_process_import(name):
        raise AssertionError(f"must not import {name} in the remote helper process")

    def run(command, **kwargs):
        commands.append(command)
        if len(command) == 4:
            return json.dumps(
                {
                    "kernelgen_server": "6.2.4",
                    "fastapi": "0.116.1",
                }
            )
        return str(module_path)

    monkeypatch.setattr(remote_server.importlib, "import_module", fail_in_process_import)
    monkeypatch.setattr(remote_server, "_run", run)

    versions = remote_server._validate_imports(["kernelgen_server", "fastapi"])
    remote_server._validate_server_import_root(kgs_root)

    assert versions == {"kernelgen_server": "6.2.4", "fastapi": "0.116.1"}
    assert len(commands) == 2
    assert all(
        command[:2] == [remote_server.sys.executable, "-c"]
        for command in commands
    )
    assert json.loads(commands[0][3]) == ["kernelgen_server", "fastapi"]


def test_remote_validation_rejects_stale_editable_checkout(tmp_path, monkeypatch):
    expected_root = tmp_path / "expected-kgs"
    stale_module = tmp_path / "stale-kgs" / "kernelgen_server" / "__init__.py"
    monkeypatch.setattr(remote_server, "_run", lambda command: str(stale_module))

    with pytest.raises(RuntimeError, match="expected checkout"):
        remote_server._validate_server_import_root(expected_root)
