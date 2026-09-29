import pytest
import json
import subprocess
import sys

from kernelgen.cli import server
from kernelgen.cli.main import build_parser


@pytest.mark.parametrize("flag", ["--install-gems", "--install_gems"])
@pytest.mark.parametrize("target", ["local", "remote"])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("first_start", [False, True])
def test_start_prepares_gems_before_start(tmp_path, monkeypatch, flag, target, configured, first_start):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path))
    args = build_parser().parse_args(["server", "start", "test", flag])
    config = {"instance": "test", "target": target}
    if not first_start:
        server._save_config("test", config)
    monkeypatch.setattr(server, "_create_instance_config", lambda *a: config)
    events = []
    monkeypatch.setattr(server, "_assert_existing_config_matches", lambda *a: None)
    monkeypatch.setattr(server, "_validate_instance_checkout", lambda *a: None)
    monkeypatch.setattr(server, "_ensure_local_environment", lambda *a: None)
    monkeypatch.setattr(server, "_configured_flaggems", lambda c: {"commit": "pinned"} if configured else None)
    monkeypatch.setattr(server, "_flaggems_spec", lambda c: {"commit": "pinned"})
    monkeypatch.setattr(server, "_install_flaggems_locked", lambda *a, **k: events.append("install"))
    monkeypatch.setattr(server, "_endpoint_lock_paths", lambda c: [])
    monkeypatch.setattr(server, "_start_configured_instance", lambda *a: events.append("start") or 0)
    assert server._command_start(args) == 0
    assert events == (["start"] if configured else ["install", "start"])


def test_install_failure_prevents_start(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path))
    args = build_parser().parse_args(["server", "start", "test", "--install-gems"])
    server._save_config("test", {"instance": "test", "target": "local"})
    monkeypatch.setattr(server, "_assert_existing_config_matches", lambda *a: None)
    monkeypatch.setattr(server, "_validate_instance_checkout", lambda *a: None)
    monkeypatch.setattr(server, "_ensure_local_environment", lambda *a: None)
    monkeypatch.setattr(server, "_configured_flaggems", lambda c: None)
    def fail(*args, **kwargs):
        raise RuntimeError("installation failed")
    monkeypatch.setattr(server, "_install_flaggems_locked", fail)
    monkeypatch.setattr(server, "_start_configured_instance", lambda *a: pytest.fail("must not start"))
    with pytest.raises(RuntimeError, match="installation failed"):
        server._command_start(args)


def test_remote_compatibility_bootstrap_reads_verified_checkout(tmp_path):
    root = tmp_path / "remote-kgs"
    root.mkdir()
    manifest = {"frameworks": {"flaggems": {"revision_policy": "exact", "revision": "a" * 40}}}
    (root / "compatibility.yaml").write_text(json.dumps(manifest))
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("remote", "add", "origin", "https://example.com/kgs.git")
    git("add", "compatibility.yaml")
    git("commit", "-qm", "fixture")
    payload = {"kgs_root": str(root), "repository": "https://example.com/kgs.git",
               "release": "v6.3.4", "commit": git("rev-parse", "HEAD")}
    command = server._remote_management_command(sys.executable, "compatibility", payload)
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert json.loads(json.loads(result.stdout.removeprefix(server.RESULT_MARKER))["text"]) == manifest
    (root / "compatibility.yaml").write_text("changed")
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_remote_gems_spec_never_reads_local_kgs(monkeypatch):
    calls = []
    def remote(config, action, additions=None):
        calls.append(action)
        return {"text": json.dumps({"frameworks": {"flaggems": {"revision_policy": "exact",
                "repository": "https://example.com/gems.git", "branch": "main", "revision": "b" * 40}}})}
    monkeypatch.setattr(server, "_remote_call_config", remote)
    monkeypatch.setattr(server, "_read_yaml", lambda *a: pytest.fail("local checkout must not be read"))
    assert server._flaggems_spec({"target": "remote", "kgs_root": "/nonexistent"})["commit"] == "b" * 40
    assert calls == ["compatibility"]
