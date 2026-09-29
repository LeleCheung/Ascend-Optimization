"""Only new experiment preparation follows Gems; KGS and old snapshots stay pinned."""
from pathlib import Path

import pytest
import yaml

from kernelgen.cli import server
from kernelgen.cli.main import build_parser
from kernelgen.tests.test_cli_server import flaggems_repository as flaggems_repository
from kernelgen.tests.test_cli_server import _remote_config, _write_compatibility


@pytest.fixture
def instance(tmp_path, monkeypatch, flaggems_repository):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    kgs = tmp_path / "kgs"
    _write_compatibility(kgs)
    path = kgs / "compatibility.yaml"
    document = yaml.safe_load(path.read_text())
    document["frameworks"]["flaggems"] = {
        "repository": str(flaggems_repository), "branch": "development", "revision_policy": "branch"}
    path.write_text(yaml.safe_dump(document))
    config = {**_remote_config("test"), "target": "local", "kgs_root": str(kgs)}
    server._save_config("test", config)
    monkeypatch.setattr(server, "_active_process", lambda _: None)
    monkeypatch.setattr(server, "_validate_instance_checkout", lambda _: None)
    return flaggems_repository, config


def test_repeated_preparation_resolves_latest_once_and_preserves_old_checkout(instance, monkeypatch):
    repository, config = instance
    kgs_commit = config["kgs_commit"]
    calls = []
    resolve = server._resolve_flaggems_revision
    def observed(*args):
        calls.append(args)
        return resolve(*args)
    monkeypatch.setattr(server, "_resolve_flaggems_revision", observed)
    assert server._flaggems_spec(config)["commit"] is None
    server._install_flaggems_locked("test", config, allow_running_noop=True)
    first = dict(config)
    server._git_value(repository, "commit", "--allow-empty", "-m", "new Gems tests")
    server._install_flaggems_locked("test", config, allow_running_noop=True)
    assert calls == [(str(repository), "development")] * 2
    assert config["flaggems_commit"] != first["flaggems_commit"]
    assert server._git_value(Path(first["flaggems_root"]), "rev-parse", "HEAD") == first["flaggems_commit"]
    assert config["kgs_commit"] == kgs_commit


def test_running_instance_can_reuse_but_not_change_its_framework(instance, monkeypatch):
    repository, config = instance
    server._install_flaggems_locked("test", config)
    before = dict(config)
    monkeypatch.setattr(server, "_active_process", lambda _: object())
    server._install_flaggems_locked("test", config, allow_running_noop=True)
    assert config == before
    server._git_value(repository, "commit", "--allow-empty", "-m", "next")
    with pytest.raises(RuntimeError, match="stop server instance"):
        server._install_flaggems_locked("test", config, allow_running_noop=True)
    assert server._load_config("test") == before


@pytest.mark.parametrize("branch,tracks", [("development", True), (None, False), ("explicit-other", False)])
def test_start_checks_default_branch_but_preserves_explicit_resume_snapshot(instance, monkeypatch, branch, tracks):
    _, config = instance
    server._install_flaggems_locked("test", config)
    config["flaggems_branch"] = branch
    server._save_config("test", config)
    events = []
    monkeypatch.setattr(server, "_assert_existing_config_matches", lambda *a: None)
    monkeypatch.setattr(server, "_ensure_local_environment", lambda *a: None)
    monkeypatch.setattr(server, "_endpoint_lock_paths", lambda *a: [])
    monkeypatch.setattr(server, "_install_flaggems_locked", lambda *a, **k: events.append("prepare"))
    monkeypatch.setattr(server, "_start_configured_instance", lambda *a: events.append("start") or 0)
    args = build_parser().parse_args(["server", "start", "test"])
    assert server._command_start(args) == 0
    assert events == (["prepare", "start"] if tracks else ["start"])


def test_network_failure_does_not_change_selection(instance, monkeypatch):
    _, config = instance
    server._install_flaggems_locked("test", config)
    before = dict(config)
    monkeypatch.setattr(server, "_resolve_flaggems_revision", lambda *a: (_ for _ in ()).throw(RuntimeError("network unavailable")))
    with pytest.raises(RuntimeError, match="network unavailable"):
        server._install_flaggems_locked("test", config, allow_running_noop=True)
    assert server._load_config("test") == before


def test_branch_policy_rejects_a_second_static_revision(instance):
    _, config = instance
    path = Path(config["kgs_root"]) / "compatibility.yaml"
    document = yaml.safe_load(path.read_text())
    document["frameworks"]["flaggems"]["revision"] = "a" * 40
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="fixed revision"):
        server._flaggems_spec(config)


@pytest.mark.parametrize("running,latest", [(False, "b" * 40), (True, "a" * 40), (True, "b" * 40)])
def test_remote_policy_never_uses_local_kgs_or_changes_a_running_framework(tmp_path, monkeypatch, running, latest):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = {**_remote_config("test"), "flaggems_root": "/remote/gems/a",
              "flaggems_commit": "a" * 40, "flaggems_branch": "kernelgen-dev"}
    before = dict(config)
    server._save_config("test", config)
    monkeypatch.setattr(server, "_active_process", lambda _: None)
    monkeypatch.setattr(server, "_validate_instance_checkout", lambda _: None)
    monkeypatch.setattr(server, "_read_yaml", lambda _: pytest.fail("must not load a local KGS manifest"))
    monkeypatch.setattr(server, "_resolve_flaggems_revision", lambda *a: (latest, "kernelgen-dev"))
    installed = []
    def remote(config, action, additions=None):
        if action == "status":
            return {"running": running}
        if action == "compatibility":
            return {"text": yaml.safe_dump({"frameworks": {"flaggems": {
                "repository": "https://github.com/flagos-ai/FlagGems.git",
                "branch": "kernelgen-dev", "revision_policy": "branch"}}})}
        assert action == "install_flaggems"
        installed.append(additions["flaggems"])
        return {"root": "/remote/gems/" + latest, "commit": latest}
    monkeypatch.setattr(server, "_remote_call_config", remote)
    if running and latest != before["flaggems_commit"]:
        with pytest.raises(RuntimeError, match="stop remote server"):
            server._install_flaggems_locked("test", config, allow_running_noop=True)
        assert config == before and not installed
    else:
        server._install_flaggems_locked("test", config, allow_running_noop=True)
        assert config["flaggems_commit"] == latest
        assert len(installed) == (0 if running else 1)
    assert config["kgs_commit"] == before["kgs_commit"]
