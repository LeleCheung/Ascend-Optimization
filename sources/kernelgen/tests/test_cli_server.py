"""Host-only tests for KGS deployment manifest handling."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kernelgen.cli import server


def test_enflame_backend_parser_and_required_runtime():
    from kernelgen.cli.main import build_parser
    for args in (["server", "start", "s60", "--backend", "enflame"],
                 ["server", "configure", "s60", "--backend", "enflame"]):
        assert build_parser().parse_args(args).backend == "enflame"
    assert "torch_gcu" in server._required_runtime_modules("enflame")
    assert "torch_gcu" not in server._required_runtime_modules("cuda")


@pytest.mark.parametrize("backend,expected", [("enflame", {"TOPS_VISIBLE_DEVICES": "7"}),
                                              ("cuda", {"CUDA_VISIBLE_DEVICES": "7"})])
def test_local_start_isolates_enflame_visibility(tmp_path, monkeypatch, backend, expected):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    for variable in ("TOPS_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.setenv(variable, "0,1,2")
    captured = {}
    class Process:
        pid = 321
    monkeypatch.setattr(server.subprocess, "Popen", lambda command, **kwargs: captured.update(command=command, **kwargs) or Process())
    monkeypatch.setattr(server, "process_start_identity", lambda pid: "start")
    monkeypatch.setattr(server, "_wait_until_ready", lambda *args, **kwargs: {})
    config = {**_remote_config(), "target": "local", "backend": backend, "devices": ["7"],
              "python": sys.executable, "kgs_root": str(tmp_path)}
    server._start_local("test", Namespace(startup_timeout=1), config)
    visible = {k: v for k, v in captured["env"].items() if k in {n for names in server.DEVICE_ENVIRONMENTS.values() for n in names}}
    assert visible == expected
    assert captured["command"][captured["command"].index("--backend") + 1] == backend


def test_install_protects_torch_gcu(tmp_path, monkeypatch):
    versions = iter([{"torch_gcu": "original"}, {"torch_gcu": "changed"}])
    monkeypatch.setattr(server, "_package_versions", lambda python: next(versions))
    monkeypatch.setattr(server, "_run", lambda *args, **kwargs: "")
    with pytest.raises(RuntimeError, match="protected runtime packages: torch_gcu"):
        server._install_server(sys.executable, tmp_path)


def test_package_snapshot_includes_torch_gcu(monkeypatch):
    def run(command):
        assert "'torch-gcu'" in command[2]
        return '{"torch_gcu": "test"}'
    monkeypatch.setattr(server, "_run", run)
    assert server._package_versions(sys.executable)["torch_gcu"] == "test"


def test_cli_bootstrap_does_not_import_kernelgen_server():
    script = """
import sys
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'kernelgen_server' or fullname.startswith('kernelgen_server.'):
            raise RuntimeError(f'unexpected bootstrap import: {fullname}')
        return None
sys.meta_path.insert(0, Blocker())
import kernelgen.cli.main
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_compatibility_reader_supports_published_and_simplified_schema():
    assert server._compatibility(
        {
            "schema_version": 1,
            "server_release": {
                "version": "v6.2.4",
                "api_version": "v6.2",
                "supported_api_versions": ["v6.0", "v6.2"],
            },
        }
    ) == ("v6.2.4", "v6.2", ["v6.0", "v6.2"])
    assert server._compatibility(
        {
            "schema_version": 2,
            "server_release": {
                "version": "v7.0.0",
                "protocol_version": "v7.0",
            },
            "supported_catalog_api_versions": ["v7.0"],
        }
    ) == ("v7.0.0", "v7.0", ["v7.0"])


def _write_compatibility(kgs_root: Path) -> str:
    revision = "d64794e63b502cb836bc015a92a62c42de4be05a"
    kgs_root.mkdir(parents=True, exist_ok=True)
    (kgs_root / "compatibility.yaml").write_text(
        f"""schema_version: 1
server_release:
  version: v6.2.4
  api_version: v6.2
frameworks:
  flaggems:
    repository: https://example.com/FlagGems.git
    branch: pinned-branch
    revision: {revision}
    revision_policy: exact
""",
        encoding="utf-8",
    )
    return revision


def test_flaggems_spec_comes_from_locked_kgs_compatibility(tmp_path):
    kgs_root = tmp_path / "kgs"
    revision = _write_compatibility(kgs_root)

    assert server._flaggems_spec({"target": "local", "kgs_root": str(kgs_root)}) == {
        "repository": "https://example.com/FlagGems.git",
        "branch": "pinned-branch",
        "commit": revision,
    }


def _start_args(**overrides):
    values = {
        "name": "local-npu",
        "target": "local",
        "backend": "npu",
        "devices": "0,1",
        "timing": "auto",
        "port": 18080,
        "remote_port": None,
        "listen_port": None,
        "max_workers": None,
        "max_streams": None,
        "ssh_command": None,
        "container": None,
        "remote_python": None,
        "remote_kgs_root": None,
        "remote_state_root": None,
        "remote_env_file": None,
        "python": sys.executable,
        "kgs_root": None,
        "startup_timeout": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_local_install_uses_declared_isolated_build_environment(tmp_path, monkeypatch):
    kgs_root = tmp_path / "kgs"
    versions = {
        "torch": "test",
        "triton": "test",
        "torch_npu": None,
        "torch_mlu": None,
        "torch_musa": None,
    }
    commands = []
    monkeypatch.setattr(server, "_package_versions", lambda python: versions)
    monkeypatch.setattr(
        server,
        "_run",
        lambda command, **kwargs: commands.append(command) or "",
    )

    server._install_server(
        "/opt/kernelgen/bin/python3",
        kgs_root,
    )

    assert commands == [
        [
            "/opt/kernelgen/bin/python3",
            "-m",
            "pip",
            "install",
            "-e",
            str(kgs_root / "client"),
        ],
        [
            "/opt/kernelgen/bin/python3",
            "-m",
            "pip",
            "install",
            "-e",
            f"{kgs_root}[server]",
        ]
    ]


@pytest.mark.parametrize("legacy_matrix", ["absent", "old-kg", "stale-entry", "matching"])
def test_first_start_configuration_persists_exact_locked_combination(tmp_path, monkeypatch, legacy_matrix):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    lock = server._read_yaml(server._lock_manifest_path())
    locked_release = lock["kgs"]["release"]
    kg_release = lock["kg_release"]
    kgs_root = tmp_path / "kgs"
    kgs_root.mkdir(parents=True)
    compatibility = {
        "schema_version": 1,
        "server_release": {
            "version": locked_release,
            "api_version": "v6.2",
            "supported_api_versions": ["v6.0", "v6.2"],
        },
    }
    if legacy_matrix != "absent":
        compatibility["kernelgen_releases"] = {
            "v0.0.1" if legacy_matrix == "old-kg" else kg_release: {
                "required_api_version": "v0.0" if legacy_matrix == "stale-entry" else "v6.2",
                "latest_validated_kgs": "v0.0.1" if legacy_matrix == "stale-entry" else locked_release,
            },
        }
    (kgs_root / "compatibility.yaml").write_text(yaml.safe_dump(compatibility), encoding="utf-8")
    checkout_calls = []
    monkeypatch.setattr(server, "_prepare_checkout", lambda root, **kwargs: checkout_calls.append((root, kwargs)))
    monkeypatch.setattr(
        server,
        "_package_versions",
        lambda python: {
            "torch": "test",
            "triton": "test",
            "torch_npu": "test",
        },
    )
    validated_modules = []
    monkeypatch.setattr(
        server,
        "_validate_imports",
        lambda python, modules: validated_modules.append(modules),
    )
    monkeypatch.setattr(
        server,
        "_validate_local_server_install",
        lambda python, root: None,
    )
    args = _start_args(kgs_root=kgs_root)

    config = server._create_instance_config(args, "local-npu")
    persisted = json.loads(server._config_path("local-npu").read_text(encoding="utf-8"))
    assert config["kg_release"] == kg_release
    assert config["kgs_release"] == locked_release
    assert config["kgs_commit"] == server._read_yaml(server._lock_manifest_path())["kgs"]["commit"]
    assert config["protocol_version"] == "v6.2"
    assert config["max_workers"] == 2
    assert config["instance"] == "local-npu"
    assert config["target"] == "local"
    assert "catalog_name" not in config
    assert "framework_root" not in config
    assert persisted == config
    assert checkout_calls == [(kgs_root, {
        "repository": lock["kgs"]["repository"],
        "commit": lock["kgs"]["commit"],
        "release": None,
        "branch": None,
    })]
    assert validated_modules == []  # Environment preparation belongs to start, not config creation.


@pytest.mark.parametrize("field", ["version", "api_version"])
def test_first_start_rejects_server_facts_that_disagree_with_kg_lock(tmp_path, monkeypatch, field):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    lock = server._read_yaml(server._lock_manifest_path())
    kgs_root = tmp_path / "kgs"
    kgs_root.mkdir()
    release = {"version": lock["kgs"]["release"], "api_version": lock["validated_protocol"]}
    release[field] = "v0.0"
    (kgs_root / "compatibility.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "server_release": release}), encoding="utf-8",
    )
    monkeypatch.setattr(server, "_prepare_checkout", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="does not match the KG deployment lock"):
        server._create_instance_config(_start_args(kgs_root=kgs_root), "local-npu")
    assert not server._config_path("local-npu").exists()


def test_remote_config_does_not_require_local_kgs_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    def forbidden(*args, **kwargs):
        pytest.fail("remote config accessed a local KGS checkout")
    monkeypatch.setattr(server, "_prepare_checkout", forbidden)
    monkeypatch.setattr(server, "_validate_checkout", forbidden)
    monkeypatch.setattr(server, "_compatibility", forbidden)
    local_checkout = tmp_path / "missing-local-kgs"
    args = _start_args(target="remote", port=None, remote_port=18080, listen_port=19080,
                       ssh_command="ssh test-host", container="target-container", kgs_root=local_checkout)
    config = server._create_instance_config(args, "remote-test")
    lock = server._read_yaml(server._lock_manifest_path())
    assert config["kgs_commit"] == lock["kgs"]["commit"]
    assert config["kgs_release"] == lock["kgs"]["release"]
    assert config["protocol_version"] == lock["validated_protocol"]
    assert not local_checkout.exists()
    server._validate_instance_checkout(config)
    payload = server._remote_payload(config)
    assert payload["commit"] == lock["kgs"]["commit"]


def test_local_instance_still_validates_exact_checkout(monkeypatch, tmp_path):
    lock = server._read_yaml(server._lock_manifest_path())
    calls = []
    monkeypatch.setattr(server, "_validate_checkout", lambda *a, **kw: calls.append((a, kw)))
    config = {"target": "local", "kg_release": server._kg_release(),
              "kgs_release": lock["kgs"]["release"], "kgs_root": str(tmp_path)}
    server._validate_instance_checkout(config)
    assert calls == [((tmp_path,), {"repository": lock["kgs"]["repository"], "commit": lock["kgs"]["commit"]})]


def test_ssh_command_is_noninteractive_and_container_is_quoted():
    ssh_command = server._parse_ssh_command(
        "ssh -p 2224 -i /tmp/deploy-key user@example.com"
    )

    command = server._remote_exec_command(
        ssh_command,
        container="kernelgen-ascend",
        command=["python3", "-u", "-", "status", "payload"],
    )

    assert command[0] == "ssh"
    assert "-T" in command
    assert "BatchMode=yes" in command
    assert "ServerAliveInterval=6" in command
    assert command[-1].startswith(
        "sudo docker exec -i kernelgen-ascend python3 -u - status"
    )


@pytest.mark.parametrize(
    "value",
    [
        "bash -c ssh host",
        "ssh -tt host",
        "ssh host\nwhoami",
    ],
)
def test_ssh_command_rejects_shells_ttys_and_newlines(value):
    with pytest.raises(ValueError):
        server._parse_ssh_command(value)


def test_remote_call_invokes_kgs_management_without_streaming_source(monkeypatch):
    captured = {}

    def run(command, *, text, capture_output):
        captured.update(command=command)
        return Namespace(
            returncode=0,
            stdout=(
                "welcome banner\n"
                f'{server.RESULT_MARKER}{{"running":true,"pid":42}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(server.subprocess, "run", run)

    result = server._remote_call(
        ssh_command=["ssh", "host"],
        container="container-name",
        remote_python="python3",
        action="status",
        payload={"state_root": "~/.kernelgen/servers/test", "kgs_root": "/remote/kgs",
                 "repository": "https://example.com/kgs.git", "release": "v6.2.4", "commit": "a" * 40},
    )

    assert result == {"running": True, "pid": 42}
    assert "kernelgen_server/management.py" in captured["command"][-1]
    assert "def _start" not in captured["command"][-1]
    assert captured["command"][-1].startswith("sudo docker exec -i container-name /bin/bash")


def test_remote_bootstrap_clones_exact_commit_and_refuses_dirty_checkout(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    management = repository / "kernelgen_server" / "management.py"
    management.parent.mkdir()
    management.write_text("print('managed by KGS')\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    git("init", "-b", "test-bootstrap")
    git("add", "--", "kernelgen_server/management.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture")
    payload = {
        "kgs_root": str(tmp_path / "deployed"), "repository": str(repository),
        "release": "test-bootstrap", "commit": git("rev-parse", "HEAD"),
    }
    command = server._remote_management_command(sys.executable, "status", payload)
    for _ in range(2):
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "managed by KGS"
    (tmp_path / "deployed" / "dirty").write_text("preserve evidence")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert "not clean" in result.stderr


def _remote_config(instance="ascend-01"):
    return {
        "schema_version": "1.0",
        "instance": instance,
        "target": "remote",
        "kg_release": "v6.1.2",
        "kgs_release": "v6.2.4",
        "kgs_commit": "9f9ae9cf40f6094fbff47c13ec14540a40e2c9d4",
        "protocol_version": "v6.2",
        "kgs_root": "/local/kgs",
        "backend": "npu",
        "devices": ["0", "1"],
        "timing": "auto",
        "port": 18080,
        "listen_port": 19080,
        "max_workers": 2,
        "max_streams": 4,
        "ssh_command": ["ssh", "remote-host"],
        "container": "kgs-container",
        "remote_python": "python3",
        "remote_kgs_root": "~/.kernelgen/kgs/v6.2.4",
        "remote_state_root": f"~/.kernelgen/servers/{instance}",
        "remote_env_file": "/opt/kernelgen/deployment.env.sh",
    }


def _remote_process(instance="ascend-01"):
    return server.ServerProcessRecord(
        instance=instance,
        target="remote",
        pid=123,
        process_start="proxy-start",
        log_path=Path("/local/remote-proxy.log"),
        server_url="http://127.0.0.1:19080",
        kgs_release="v6.2.4",
        kgs_commit="commit",
        protocol_version="v6.2",
        backend="npu",
        devices=["0", "1"],
        port=18080,
        max_workers=2,
        remote_pid=321,
        remote_process_start="remote-start",
        remote_log_path="/remote/kernelgen-server.log",
    )


def test_remote_payload_ignores_legacy_catalog_binding():
    config = {
        **_remote_config(),
        "catalog_name": "flaggems-adapter-definitions",
        "catalog_evaluator": "flaggems",
        "remote_framework_base": "/legacy/frameworks",
    }

    payload = server._remote_payload(config)

    assert "catalog_name" not in payload
    assert "framework_base" not in payload


def test_remote_payload_includes_machine_local_deployment_env_file():
    payload = server._remote_payload(_remote_config())

    assert payload["env_file"] == "/opt/kernelgen/deployment.env.sh"


def test_remote_payload_keeps_existing_instances_without_deployment_env_compatible():
    config = _remote_config()
    config.pop("remote_env_file")

    assert "env_file" not in server._remote_payload(config)


@pytest.mark.parametrize("value", ["", "first\nsecond", "bad\0path"])
def test_remote_env_file_rejects_empty_or_multiline_values(value):
    with pytest.raises(ValueError, match="remote-env-file"):
        server._validate_remote_env_file(value)


def test_remote_payload_includes_installed_flaggems_checkout(tmp_path, monkeypatch):
    kgs_root = tmp_path / "kgs"
    revision = _write_compatibility(kgs_root)
    config = {
        **_remote_config(),
        "kgs_root": str(kgs_root),
        "flaggems_root": "/remote/FlagGems",
        "flaggems_commit": revision,
    }

    monkeypatch.setattr(server, "_remote_call_config", lambda *a: {"text": (kgs_root / "compatibility.yaml").read_text()})
    payload = server._remote_payload(config, include_flaggems=True)

    assert payload["flaggems"] == {
        "repository": "https://example.com/FlagGems.git",
        "branch": "pinned-branch",
        "commit": revision,
        "root": "/remote/FlagGems",
    }


def _healthy_status():
    return {
        "api_version": "v6.2",
        "server_version": "v6.2.4",
        "backend": "npu",
        "workers": 2,
        "devices": ["0", "1"],
        "scheduler": {
            "device_slots": 2,
            "healthy": 2,
            "checking": 0,
            "broken": 0,
            "max_active": 2,
            "active": 0,
            "waiting": 0,
            "available": 2,
        },
    }


def test_live_status_does_not_bind_server_to_legacy_catalog_evaluator():
    config = {**_remote_config(), "catalog_evaluator": "flaggems"}

    assert server._live_status_problems(
        config,
        _healthy_status(),
        require_idle=True,
    ) == []


def test_live_status_requires_flaggems_capability_only_after_installation():
    config = {**_remote_config(), "flaggems_root": "/remote/FlagGems"}
    status = _healthy_status()

    assert server._live_status_problems(
        config,
        status,
        require_idle=True,
    ) == ["running KGS does not provide the installed FlagGems adapter"]

    status["evaluation_adapters"] = ["native", "flaggems"]
    assert server._live_status_problems(
        config,
        status,
        require_idle=True,
    ) == []


def test_remote_deployment_leaves_install_decision_to_kgs(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = _remote_config()
    server._save_config("ascend-01", config)
    calls = []

    def remote_call(config, action, additions=None):
        calls.append((action, additions))
        return {
            "pid": 321,
            "process_start": "remote-start",
            "log_path": "/remote/kernelgen-server.log",
        }

    monkeypatch.setattr(server, "_remote_call_config", remote_call)

    server._start_remote_server("ascend-01", config)
    server._start_remote_server("ascend-01", config)

    assert calls == [("start", None), ("start", None)]
    assert "remote_deployed" not in server._load_config("ascend-01")


def test_remote_commands_use_recorded_commit_not_a_new_deployment_lock(monkeypatch):
    config = _remote_config()
    monkeypatch.setattr(server, "_read_yaml", lambda path: {
        "kgs": {"repository": "git@gitee.example:team/kgs.git", "release": "new-release", "commit": "new-commit"},
    })
    payload = server._remote_payload(config)
    assert payload["commit"] == config["kgs_commit"]
    assert payload["release"] == config["kgs_release"]


def test_remote_proxy_records_named_local_and_remote_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = _remote_config()

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(server.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(server, "process_start_identity", lambda pid: "proxy-start")
    monkeypatch.setattr(server, "_wait_for_proxy_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_wait_until_ready", lambda *args, **kwargs: {})
    args = Namespace(startup_timeout=1)
    remote_result = {
        "pid": 321,
        "process_start": "remote-start",
        "log_path": "/remote/kernelgen-server.log",
    }

    assert server._start_remote_proxy("ascend-01", args, config, remote_result) == 0

    record = server._load_process("ascend-01")
    assert record is not None
    assert record.instance == "ascend-01"
    assert record.target == "remote"
    assert record.pid == 123
    assert record.remote_pid == 321
    assert record.server_url == "http://127.0.0.1:19080"
    assert Path(record.log_path).name == "remote-proxy.log"


@pytest.mark.parametrize("previous_run", [False, True])
def test_remote_proxy_cannot_borrow_occupied_port_health(tmp_path, monkeypatch, previous_run):
    import socket
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = _remote_config()
    if previous_run:
        server.atomic_write_json(server._process_path("occupied"),
                                 _remote_process("occupied").model_dump(mode="json"))
    calls = []
    monkeypatch.setattr(server, "_http_status", lambda config: calls.append(config) or _healthy_status())
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        config["listen_port"] = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="failed before readiness"):
            server._start_remote_proxy("occupied", Namespace(startup_timeout=5), config,
                                       {"pid": 654, "process_start": "remote", "log_path": "/remote/log"})
        assert not calls
        record = server._load_process("occupied")
        if previous_run:
            assert record.remote_pid == 654
            assert record.remote_process_start == "remote"
            assert record.pid == 123
            stop_calls = []
            monkeypatch.setattr(server, "_load_config", lambda instance: config)
            monkeypatch.setattr(server, "process_is_alive", lambda *args: False)
            monkeypatch.setattr(server, "_remote_call_config", lambda config, action, payload:
                                stop_calls.append((action, payload)) or {})
            assert server._command_stop(Namespace(name="occupied", timeout=1)) == 0
            assert stop_calls == [("stop", {"timeout": 1, "expected_pid": 654,
                                            "expected_process_start": "remote"})]
        else:
            assert record is None
        assert listener.fileno() >= 0


@pytest.mark.parametrize("message", [b"READY\n", b"", b"WRONG\n"])
def test_proxy_readiness_requires_child_notification(message):
    import os
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, message)
        os.close(write_fd)
        process = Namespace(poll=lambda: None)
        if message == b"READY\n":
            server._wait_for_proxy_ready(read_fd, process, 1)
        else:
            with pytest.raises(RuntimeError, match="failed before readiness"):
                server._wait_for_proxy_ready(read_fd, process, 1)
    finally:
        os.close(read_fd)


def test_proxy_readiness_timeout():
    import os
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            server._wait_for_proxy_ready(read_fd, Namespace(poll=lambda: None), 0)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_remote_status_detects_stopped_proxy_without_exposing_ssh_command(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = _remote_config()
    server._save_config("ascend-01", config)
    record = server.ServerProcessRecord(
        instance="ascend-01",
        target="remote",
        pid=123,
        process_start="proxy-start",
        log_path=tmp_path / "proxy.log",
        server_url="http://127.0.0.1:19080",
        kgs_release="v6.2.4",
        kgs_commit="commit",
        protocol_version="v6.2",
        backend="npu",
        devices=["0"],
        port=18080,
        max_workers=1,
        remote_pid=321,
        remote_process_start="remote-start",
    )
    server.atomic_write_json(
        server._process_path("ascend-01"),
        record.model_dump(mode="json"),
    )
    monkeypatch.setattr(server, "process_is_alive", lambda *args: False)
    monkeypatch.setattr(
        server,
        "_remote_call_config",
        lambda *args, **kwargs: {"running": True, "pid": 321},
    )

    status = server._server_status("ascend-01")

    assert status["state"] == "UNREACHABLE"
    assert status["remote"]["running"] is True
    assert "ssh_command" not in status["process"]
    assert "ssh_command" not in status["config"]


def test_named_instances_keep_independent_state(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    first = _remote_config("ascend-01")
    second = _remote_config("ascend-02")
    second["listen_port"] = 19081
    server._save_config("ascend-01", first)
    server._save_config("ascend-02", second)

    assert server._load_config("ascend-01")["listen_port"] == 19080
    assert server._load_config("ascend-02")["listen_port"] == 19081
    assert server._server_root("ascend-01") != server._server_root("ascend-02")


def test_start_restores_proxy_when_remote_is_already_running(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = _remote_config()
    server._save_config("ascend-01", config)
    monkeypatch.setattr(server, "_validate_instance_checkout", lambda config: None)
    monkeypatch.setattr(server, "_endpoint_conflict", lambda *args: None)
    monkeypatch.setattr(server, "_active_process", lambda instance: None)
    monkeypatch.setattr(
        server,
        "_remote_call_config",
        lambda config, action, additions=None: {
            "running": True,
            "pid": 321,
            "process_start": "remote-start",
            "log_path": "/remote/kernelgen-server.log",
        },
    )
    recovered = []
    monkeypatch.setattr(
        server,
        "_start_remote_proxy",
        lambda instance, args, config, result: recovered.append(result) or 0,
    )
    monkeypatch.setattr(
        server,
        "_start_remote_server",
        lambda *args: pytest.fail("remote server must not be restarted"),
    )
    args = _start_args(
        name="ascend-01",
        target=None,
        backend=None,
        devices=None,
        timing=None,
        port=None,
        python=None,
        kgs_root=None,
    )

    assert server._command_start(args) == 0
    assert recovered[0]["pid"] == 321


def test_start_is_idempotent_when_remote_and_proxy_are_running(monkeypatch, capsys):
    config = _remote_config()
    process = _remote_process()
    monkeypatch.setattr(server, "_endpoint_conflict", lambda *args: None)
    monkeypatch.setattr(server, "_active_process", lambda instance: process)
    monkeypatch.setattr(
        server,
        "_remote_call_config",
        lambda *args, **kwargs: {"running": True, "pid": 321},
    )
    monkeypatch.setattr(server, "_http_status", lambda config: _healthy_status())
    monkeypatch.setattr(
        server,
        "_start_remote_proxy",
        lambda *args: pytest.fail("proxy must not be restarted"),
    )

    assert server._start_configured_instance(
        "ascend-01", Namespace(startup_timeout=1), config
    ) == 0
    assert "status: ALREADY_RUNNING" in capsys.readouterr().out


def test_start_restarts_both_when_remote_stopped_but_proxy_is_stale(monkeypatch):
    config = _remote_config()
    process = _remote_process()
    monkeypatch.setattr(server, "_endpoint_conflict", lambda *args: None)
    monkeypatch.setattr(server, "_active_process", lambda instance: process)
    monkeypatch.setattr(
        server,
        "_remote_call_config",
        lambda *args, **kwargs: {"running": False},
    )
    terminated = []
    monkeypatch.setattr(
        server,
        "_terminate_managed_process",
        lambda record: terminated.append(record.pid),
    )
    remote_result = {
        "pid": 654,
        "process_start": "new-remote-start",
        "log_path": "/remote/kernelgen-server.log",
    }
    monkeypatch.setattr(
        server,
        "_start_remote_server",
        lambda instance, config: remote_result,
    )
    restarted = []
    monkeypatch.setattr(
        server,
        "_start_remote_proxy",
        lambda instance, args, config, result: restarted.append(result) or 0,
    )

    assert server._start_configured_instance(
        "ascend-01", Namespace(startup_timeout=1), config
    ) == 0
    assert terminated == [123]
    assert restarted == [remote_result]


def test_fleet_owned_start_requires_running_fleet_daemon(monkeypatch):
    config = _remote_config()
    config["proxy_owner"] = "fleet"
    monkeypatch.setattr(server, "_endpoint_conflict", lambda *args: None)
    monkeypatch.setattr(server, "_active_process", lambda _instance: None)
    monkeypatch.setattr(
        server,
        "_remote_call_config",
        lambda *args, **kwargs: {"running": True, "pid": 321},
    )
    monkeypatch.setattr(server, "_notify_fleet_daemon", lambda: False)

    with pytest.raises(RuntimeError, match="Fleet daemon is not running"):
        server._start_configured_instance(
            "ascend-01", Namespace(startup_timeout=1), config
        )


def test_fleet_owned_start_reports_waiting_for_proxy(monkeypatch, capsys):
    config = _remote_config()
    config["proxy_owner"] = "fleet"
    monkeypatch.setattr(server, "_endpoint_conflict", lambda *args: None)
    monkeypatch.setattr(server, "_active_process", lambda _instance: None)
    monkeypatch.setattr(
        server,
        "_remote_call_config",
        lambda *args, **kwargs: {"running": True, "pid": 321},
    )
    monkeypatch.setattr(server, "_notify_fleet_daemon", lambda: True)

    assert server._start_configured_instance(
        "ascend-01", Namespace(startup_timeout=1), config
    ) == 0
    output = capsys.readouterr().out
    assert "status: WAITING_FOR_FLEET" in output
    assert "status: RUNNING" not in output


def test_local_start_exports_installed_flaggems_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    kgs_root = tmp_path / "kgs"
    flaggems_root = tmp_path / "FlagGems"
    revision = _write_compatibility(kgs_root)
    config = {
        **_remote_config("h800"),
        "target": "local",
        "kgs_root": str(kgs_root),
        "flaggems_root": str(flaggems_root),
        "flaggems_commit": revision,
        "python": sys.executable,
    }
    captured = {}

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda command, **kwargs: captured.update(kwargs) or Process(),
    )
    monkeypatch.setattr(server, "process_start_identity", lambda pid: "start-id")
    monkeypatch.setattr(server, "_wait_until_ready", lambda *args, **kwargs: {})

    assert server._start_local("h800", Namespace(startup_timeout=1), config) == 0
    assert captured["env"]["KGS_FLAGGEMS_ROOT"] == str(flaggems_root)


def test_server_list_keeps_remote_state_local_and_fast(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    server._save_config("ascend-01", _remote_config())
    local = {**_remote_config("local-npu"), "target": "local", "listen_port": 18081}
    server._save_config("local-npu", local)
    monkeypatch.setattr(server, "_active_process", lambda instance: None)
    monkeypatch.setattr(
        server,
        "_remote_call_config",
        lambda *args, **kwargs: pytest.fail("server list must not open SSH"),
    )

    assert server._command_list(Namespace(json=True)) == 0
    result = json.loads(capsys.readouterr().out)
    assert [(item["name"], item["state"]) for item in result["servers"]] == [
        ("ascend-01", "REMOTE_UNKNOWN"),
        ("local-npu", "STOPPED"),
    ]


def test_configure_updates_only_a_stopped_named_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = {
        **_remote_config("local-npu"),
        "target": "local",
        "port": 18080,
        "listen_port": 18080,
    }
    server._save_config("local-npu", config)
    monkeypatch.setattr(server, "_active_process", lambda instance: None)
    args = Namespace(
        name="local-npu",
        backend=None,
        devices="2,3",
        timing=None,
        port=18081,
        remote_port=None,
        listen_port=None,
        max_workers=4,
        max_streams=None,
        ssh_command=None,
        container=None,
        remote_python=None,
        remote_kgs_root=None,
        remote_state_root=None,
        remote_env_file=None,
    )

    assert server._command_configure(args) == 0
    updated = server._load_config("local-npu")
    assert updated["devices"] == ["2", "3"]
    assert updated["port"] == 18081
    assert updated["listen_port"] == 18081
    assert updated["max_workers"] == 4


def test_configure_updates_remote_env_file_and_reenables_deployment(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = _remote_config()
    server._save_config("ascend-01", config)
    monkeypatch.setattr(server, "_active_process", lambda instance: None)
    monkeypatch.setattr(
        server,
        "_remote_call_config",
        lambda *args, **kwargs: {"running": False},
    )
    args = Namespace(
        name="ascend-01",
        backend=None,
        devices=None,
        timing=None,
        port=None,
        remote_port=None,
        listen_port=None,
        max_workers=None,
        max_streams=None,
        ssh_command=None,
        container=None,
        remote_python=None,
        remote_kgs_root=None,
        remote_state_root=None,
        remote_env_file="/data/kernelgen/deployment.env.sh",
    )

    assert server._command_configure(args) == 0

    updated = server._load_config("ascend-01")
    assert updated["remote_env_file"] == "/data/kernelgen/deployment.env.sh"
    assert "remote_deployed" not in updated


def test_install_flaggems_prepares_local_checkout_and_persists_it(
    tmp_path,
    monkeypatch,
    capsys,
):
    cli_state = tmp_path / "state"
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(cli_state))
    kgs_root = tmp_path / "kgs"
    revision = _write_compatibility(kgs_root)
    config = {
        **_remote_config("h800"),
        "target": "local",
        "kgs_root": str(kgs_root),
    }
    server._save_config("h800", config)
    monkeypatch.setattr(server, "_active_process", lambda instance: None)
    monkeypatch.setattr(server, "_validate_instance_checkout", lambda config: None)
    prepared = []
    monkeypatch.setattr(
        server,
        "_prepare_checkout",
        lambda root, **options: prepared.append((root, options)),
    )

    assert server._command_install_flaggems(Namespace(name="h800")) == 0

    expected_root = cli_state / "frameworks" / "FlagGems" / revision
    updated = server._load_config("h800")
    assert prepared == [
        (
            expected_root,
            {
                "repository": "https://example.com/FlagGems.git",
                "release": None,
                "branch": "pinned-branch",
                "commit": revision,
            },
        )
    ]
    assert updated["flaggems_root"] == str(expected_root)
    assert updated["flaggems_commit"] == revision
    assert "status: INSTALLED" in capsys.readouterr().out


@pytest.mark.parametrize("selected", [None, "a" * 40])
def test_install_flaggems_prepares_remote_checkout_and_persists_resolved_root(
    tmp_path,
    monkeypatch,
    selected,
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    kgs_root = tmp_path / "kgs"
    revision = _write_compatibility(kgs_root)
    revision = selected or revision
    branch = None if selected else "pinned-branch"
    config = {**_remote_config("h800"), "kgs_root": str(kgs_root)}
    server._save_config("h800", config)
    monkeypatch.setattr(server, "_active_process", lambda instance: None)
    monkeypatch.setattr(server, "_validate_instance_checkout", lambda config: None)
    calls = []

    def remote_call(config, action, additions=None):
        calls.append((action, additions))
        if action == "status":
            return {"running": False}
        if action == "compatibility":
            return {"text": (kgs_root / "compatibility.yaml").read_text()}
        return {"root": "/remote/FlagGems", "commit": revision}

    monkeypatch.setattr(server, "_remote_call_config", remote_call)

    assert server._command_install_flaggems(Namespace(name="h800", revision=selected)) == 0

    assert calls == [
        ("status", None),
        ("compatibility", None),
        (
            "install_flaggems",
            {
                "flaggems": {
                    "repository": "https://example.com/FlagGems.git",
                    "branch": branch,
                    "commit": revision,
                    "root": f"~/.kernelgen/frameworks/FlagGems/{revision}",
                }
            },
        ),
    ]
    updated = server._load_config("h800")
    assert updated["flaggems_root"] == "/remote/FlagGems"
    assert updated["flaggems_commit"] == revision
    assert updated["flaggems_branch"] == branch
    assert server._remote_payload(updated, include_flaggems=True)["flaggems"] == {
        "repository": "https://example.com/FlagGems.git",
        "branch": branch,
        "commit": revision,
        "root": "/remote/FlagGems",
    }


@pytest.fixture
def flaggems_repository(tmp_path):
    repository = tmp_path / "upstream"
    server._run(["git", "init", "--initial-branch=development", str(repository)])
    server._git_value(repository, "config", "user.name", "Test")
    server._git_value(repository, "config", "user.email", "test@example.com")
    for name in (
        "src/flag_gems/__init__.py", "benchmark/base.py", "benchmark/conftest.py",
        "benchmark/test_add.py", "tests/test_add.py",
    ):
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    server._git_value(repository, "add", "--", "src", "benchmark", "tests")
    server._git_value(repository, "commit", "-m", "initial")
    return repository


def test_explicit_flaggems_update_pins_once_and_preserves_previous_checkout(
    tmp_path, monkeypatch, flaggems_repository,
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    kgs_root = tmp_path / "kgs"
    default_revision = _write_compatibility(kgs_root)
    compatibility_path = kgs_root / "compatibility.yaml"
    compatibility = server._read_yaml(compatibility_path)
    compatibility["frameworks"]["flaggems"]["repository"] = str(flaggems_repository)
    compatibility_path.write_text(yaml.safe_dump(compatibility), encoding="utf-8")
    server._save_config("h800", {
        **_remote_config("h800"), "target": "local", "kgs_root": str(kgs_root),
    })
    monkeypatch.setattr(server, "_active_process", lambda instance: None)
    monkeypatch.setattr(server, "_validate_instance_checkout", lambda config: None)

    first_commit = server._git_value(flaggems_repository, "rev-parse", "HEAD")
    assert first_commit != default_revision
    assert server._command_install_flaggems(Namespace(name="h800", revision="development")) == 0
    first_config = server._load_config("h800")
    first_root = Path(first_config["flaggems_root"])
    assert first_config["flaggems_commit"] == first_commit
    assert first_config["flaggems_branch"] == "development"

    server._git_value(flaggems_repository, "commit", "--allow-empty", "-m", "update")
    second_commit = server._git_value(flaggems_repository, "rev-parse", "HEAD")
    # Restart/reinstall without an explicit selection must never follow branch HEAD.
    assert server._configured_flaggems(first_config)["commit"] == first_commit
    with monkeypatch.context() as no_resolution:
        no_resolution.setattr(server, "_resolve_flaggems_revision", lambda *args: pytest.fail("unexpected resolution"))
        assert server._command_install_flaggems(Namespace(name="h800")) == 0
    assert server._load_config("h800") == first_config

    assert server._command_install_flaggems(Namespace(name="h800", revision="development")) == 0
    updated = server._load_config("h800")
    assert updated["flaggems_commit"] == second_commit
    assert updated["flaggems_root"] != str(first_root)
    assert server._git_value(first_root, "rev-parse", "HEAD") == first_commit
    second_root = Path(updated["flaggems_root"])
    assert server._git_value(second_root, "rev-parse", "HEAD") == second_commit

    # Failed selection and dirty-checkout validation leave the saved selection intact.
    with pytest.raises(RuntimeError, match="command failed"):
        server._command_install_flaggems(Namespace(name="h800", revision="missing-branch"))
    assert server._load_config("h800") == updated
    (second_root / "tests/test_add.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted changes"):
        server._command_install_flaggems(Namespace(name="h800"))
    assert server._load_config("h800") == updated

    # A full commit can explicitly restore the previous snapshot without branch lookup.
    assert server._command_install_flaggems(Namespace(name="h800", revision=first_commit)) == 0
    restored = server._configured_flaggems(server._load_config("h800"))
    assert restored["root"] == str(first_root)
    assert restored["commit"] == first_commit
    assert restored["branch"] is None


def test_full_flaggems_commit_needs_no_ref_lookup(monkeypatch):
    monkeypatch.setattr(server, "_run", lambda *args, **kwargs: pytest.fail("unexpected git lookup"))
    assert server._resolve_flaggems_revision("unused", "a" * 40) == ("a" * 40, None)


@pytest.mark.parametrize("revision", ["", "--help", "bad branch", "bad\x00branch", "../bad"])
def test_invalid_flaggems_branch_is_rejected_before_network_lookup(monkeypatch, revision):
    original_run = server._run

    def local_only(command, **kwargs):
        assert command[:2] == ["git", "check-ref-format"]
        return original_run(command, **kwargs)

    monkeypatch.setattr(server, "_run", local_only)
    with pytest.raises((ValueError, RuntimeError)):
        server._resolve_flaggems_revision("unused", revision)


@pytest.mark.parametrize("selection", ["branch", "commit"])
def test_selected_flaggems_checkout_works_with_kgs_but_keeps_catalog_pin(
    tmp_path, monkeypatch, flaggems_repository, selection,
):
    from kernelgen_server import management
    from kernelgen_server.evaluation.adapters.flaggems.discovery import discover_assets

    first_commit = server._git_value(flaggems_repository, "rev-parse", "HEAD")
    server._git_value(flaggems_repository, "commit", "--allow-empty", "-m", "update")
    head = server._git_value(flaggems_repository, "rev-parse", "HEAD")
    revision = "development" if selection == "branch" else head
    commit, branch = server._resolve_flaggems_revision(str(flaggems_repository), revision)
    checkout = tmp_path / "installed"
    spec = {"repository": str(flaggems_repository), "commit": commit, "branch": branch, "root": str(checkout)}
    management._prepare_checkout(checkout, repository=spec["repository"], release=None, branch=branch, commit=commit)
    assert management._flaggems_checkout({"flaggems": spec}) == checkout
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(checkout))
    assert discover_assets("add", commit).revision == commit
    with pytest.raises(RuntimeError, match="HEAD does not match"):
        discover_assets("add", first_commit)


def test_install_flaggems_requires_a_stopped_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    server._save_config("h800", _remote_config("h800"))
    monkeypatch.setattr(server, "_active_process", lambda instance: _remote_process(instance))

    with pytest.raises(RuntimeError, match="stop server instance"):
        server._command_install_flaggems(Namespace(name="h800"))


def test_remote_instances_cannot_share_an_active_device(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    first = _remote_config("ascend-01")
    second = _remote_config("ascend-02")
    second["listen_port"] = 19081
    second["port"] = 18081
    server._save_config("ascend-01", first)
    server.atomic_write_json(
        server._process_path("ascend-01"),
        _remote_process("ascend-01").model_dump(mode="json"),
    )
    monkeypatch.setattr(server, "process_is_alive", lambda *args: True)

    assert server._endpoint_conflict("ascend-02", second) == "ascend-01"


def test_fleet_owned_remote_conflict_checks_remote_without_local_process(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    first = {
        **_remote_config("ascend-01"),
        "proxy_owner": "fleet",
        "fleet_device": "ascend",
        "fleet_deployment_status": "active",
    }
    second = _remote_config("ascend-02")
    second["listen_port"] = 19081
    second["port"] = 18081
    server._save_config("ascend-01", first)
    calls = []

    def remote_status(config, action, additions=None):
        calls.append((config["instance"], action, additions))
        return {"running": True}

    monkeypatch.setattr(server, "_remote_call_config", remote_status)

    assert server._endpoint_conflict("ascend-02", second) == "ascend-01"
    assert calls == [("ascend-01", "status", None)]


@pytest.mark.parametrize("name", ["../server", "bad/name", "-leading", ""])
def test_server_name_rejects_path_traversal(name):
    with pytest.raises(ValueError, match="server name"):
        server._validate_instance_name(name)


def test_kgs_version_listing_puts_lock_first_without_exposing_commits(monkeypatch):
    lock = server._kgs_lock()
    monkeypatch.setattr(
        server,
        "_run",
        lambda command, **_kwargs: "\n".join(
            [
                f"{'b' * 40}\trefs/heads/development",
                f"{'c' * 40}\trefs/heads/release-v7",
            ]
        ),
    )

    versions = server.list_kgs_versions()

    assert versions[0] == {
        "name": lock["kgs"]["release"],
        "recommended": True,
    }
    assert {item["name"] for item in versions[1:]} == {
        "development",
        "release-v7",
    }
    assert "commit" not in json.dumps(versions)
    assert "revision" not in json.dumps(versions)


def test_kgs_version_listing_falls_back_to_recommended_when_remote_is_unavailable(
    monkeypatch,
):
    lock = server._kgs_lock()
    monkeypatch.setattr(
        server,
        "_run_git_remote",
        lambda _command: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    assert server.list_kgs_versions() == [
        {"name": lock["kgs"]["release"], "recommended": True}
    ]


def test_kgs_version_lookup_is_noninteractive_and_bounded(monkeypatch):
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return f"{'b' * 40}\trefs/heads/development"

    monkeypatch.setattr(server, "_run", run)

    server.list_kgs_versions()

    assert captured["command"][:3] == ["git", "ls-remote", "--heads"]
    assert captured["timeout"] == server._GIT_REMOTE_TIMEOUT_SECONDS
    assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_git_remote_timeout_is_sanitized(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("git", 20)

    monkeypatch.setattr(server.subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match=r"command timed out \(git ls-remote\)"):
        server._run_git_remote(["git", "ls-remote", "private-repository"])


def test_selected_kgs_branch_is_resolved_once_and_pinned_internally(monkeypatch):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["git", "check-ref-format"]:
            return ""
        return f"{'d' * 40}\trefs/heads/development"

    monkeypatch.setattr(server, "_run", run)

    resolved = server._resolve_kgs_version("development")

    assert resolved["version"] == "development"
    assert resolved["branch"] == "development"
    assert resolved["commit"] == "d" * 40
    assert resolved["strict_release"] is False
    assert calls[-1][-1] == "refs/heads/development"


@pytest.mark.parametrize("version", ["", "--help", "bad branch", "bad\x00branch", "../bad"])
def test_invalid_kgs_branch_is_rejected_before_remote_lookup(monkeypatch, version):
    lock = server._kgs_lock()
    if version == "":
        assert server._resolve_kgs_version(version)["version"] == lock["kgs"]["release"]
        return
    original_run = server._run

    def local_only(command, **kwargs):
        assert command[:2] == ["git", "check-ref-format"]
        return original_run(command, **kwargs)

    monkeypatch.setattr(server, "_run", local_only)
    with pytest.raises((ValueError, RuntimeError)):
        server._resolve_kgs_version(version)


def test_remote_selected_kgs_branch_keeps_commit_private_in_config(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    resolved = {
        "version": "development",
        "release": "development",
        "branch": "development",
        "commit": "e" * 40,
        "repository": "git@example.test:team/kgs.git",
        "strict_release": False,
    }
    args = _start_args(
        target="remote",
        port=None,
        remote_port=18310,
        listen_port=18110,
        ssh_command="ssh test-host",
        container="target-container",
        kgs_version="development",
        resolved_kgs=resolved,
    )

    config = server._create_instance_config(args, "remote-development")

    assert config["kgs_version"] == "development"
    assert config["kgs_commit"] == "e" * 40
    assert config["strict_kgs_release"] is False
    assert config["remote_kgs_root"].endswith("e" * 40)
    status = _healthy_status()
    status["server_version"] = "v99.0.0"
    assert server._live_status_problems(config, status, require_idle=True) == []


def test_server_parser_has_named_start_and_no_init_command():
    from kernelgen.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["server", "start", "ascend-01"])

    assert args.name == "ascend-01"
    assert not hasattr(args, "catalog_name")
    assert not hasattr(args, "remote_framework_base")
    assert args.remote_env_file is None
    assert args.kgs_version is None
    selected_args = parser.parse_args(
        ["server", "start", "ascend-dev", "--kgs-version", "development"]
    )
    assert selected_args.kgs_version == "development"
    install_args = parser.parse_args(["server", "install-flaggems", "h800"])
    assert install_args.name == "h800"
    assert install_args.handler is server._command_install_flaggems
    assert install_args.revision is None
    assert parser.parse_args(["server", "install-flaggems", "h800", "--revision", "development"]).revision == "development"
    with pytest.raises(SystemExit):
        parser.parse_args(["server", "init"])
