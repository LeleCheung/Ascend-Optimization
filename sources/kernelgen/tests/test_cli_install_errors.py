import subprocess

import pytest

from kernelgen.cli import server


def test_install_error_preserves_missing_backend(monkeypatch):
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 2, "", "ModuleNotFoundError: No module named 'setuptools'"))
    with pytest.raises(RuntimeError, match="No module named 'setuptools'"):
        server._run(["python", "-m", "pip"])


def test_error_summary_redacts_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "example-test-secret")
    result = server._command_error_summary("ERROR: https://user:pass@example.test/path?token=abc example-test-secret password=hello Bearer other-secret")
    for secret in ("user:pass", "token=abc", "example-test-secret", "hello", "other-secret"):
        assert secret not in result
    assert "ERROR:" in result
    assert len(server._command_error_summary("ERROR: " + "x" * 4000)) <= 1000
    assert "BEGIN" not in server._command_error_summary("ERROR: -----BEGIN PRIVATE KEY-----")
    assert "unknown-token" not in server._command_error_summary("ERROR: Authorization: Bearer unknown-token")
    assert "two words" not in server._command_error_summary('ERROR: password="two words"')
