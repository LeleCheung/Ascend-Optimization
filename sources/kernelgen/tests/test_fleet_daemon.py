"""Tests for remote KGS discovery and filesystem-backed proxy reconciliation."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from kernelgen.service import fleet_daemon as fleet


def _config(tmp_path: Path, inventory_text: str, **overrides) -> fleet.FleetConfig:
    inventory = tmp_path / "hosts.conf"
    inventory.write_text(inventory_text, encoding="utf-8")
    identity = tmp_path / "id_ed25519"
    identity.write_text("test key", encoding="utf-8")
    values = {
        "inventory": inventory,
        "identity": identity,
        "bastion": "bastion.aiops.baai.ac.cn",
        "jump_user": "",
        "interval": 30.0,
        "probe_timeout": 5.0,
        "max_parallel": 3,
        "local_port_start": 18100,
        "local_port_end": 18120,
        "missing_grace_scans": 2,
    }
    values.update(overrides)
    return fleet.FleetConfig(**values)


def _record(
    name="musa",
    address="10.1.2.3",
    port=18303,
    login="operator@secure@10.1.2.3",
    devices=None,
):
    fields = [
        name,
        "jump",
        address,
        "2224",
        f"container_{name}",
        "/workspace/kg",
        str(port),
    ]
    if devices is not None:
        fields.append(devices)
    fields.append(login)
    return "|".join(fields) + "\n"


def _target(config: fleet.FleetConfig) -> fleet.InventoryTarget:
    return fleet.load_inventory(config)[0]


def _probe(
    target: fleet.InventoryTarget,
    *,
    state="ok",
    port=18403,
    status="ok",
    pid=123,
    max_workers=2,
    backend="musa",
    api_version="v6.2",
    target_device="S5000",
    candidate_admission=True,
    evaluation_binding=True,
    scheduler_healthy=True,
):
    return fleet.DeviceProbe(
        target=target,
        state=state,
        processes=(
            fleet.ProbeProcess(
                pid=pid,
                port=port,
                backend=backend,
                max_workers=max_workers,
                status=status,
                server_version="v6.3.4",
                api_version=api_version,
                target_device=target_device,
                candidate_admission=candidate_admission,
                evaluation_binding=evaluation_binding,
                scheduler_healthy=scheduler_healthy,
            ),
        )
        if pid
        else (),
    )


def test_inventory_parses_plain_and_legacy_jump_logins(tmp_path):
    inventory = _record() + _record(
        name="ascend",
        address="10.0.0.9",
        port=18306,
        login=(
            "ssh operator#root#e5bc9c71-b642-459e-95d1-e4dbb8c08faf"
            "@bastion.aiops.baai.ac.cn -p 2224"
        ),
    )
    targets = fleet.load_inventory(_config(tmp_path, inventory))
    assert [target.name for target in targets] == ["musa", "ascend"]
    assert targets[0].jump_login == "operator@secure@10.1.2.3"
    assert targets[1].jump_login.startswith("operator#root#")


def test_inventory_jump_user_override_matches_proxy_behavior(tmp_path):
    config = _config(tmp_path, _record(), jump_user="gupan")
    assert _target(config).jump_login == "gupan@secure@10.1.2.3"


def test_inventory_parses_explicit_devices(tmp_path):
    target = _target(_config(tmp_path, _record(devices="0,1,3")))
    assert target.devices == ("0", "1", "3")


def test_inventory_rejects_duplicate_devices(tmp_path, capsys):
    targets = fleet.load_inventory(_config(tmp_path, _record(devices="0,1,0")))
    assert targets == ()
    assert "duplicate device token" in capsys.readouterr().err


def test_inventory_accepts_new_device_names(tmp_path):
    config = _config(tmp_path, _record(name="cambricon"))
    target = _target(config)
    assert target.name == "cambricon"
    assert target.target_hardware == "cambricon"


@pytest.mark.parametrize(
    "login",
    [
        "ssh user@other.example -p 2224",
        "ssh user@bastion.aiops.baai.ac.cn -p 2200",
        "ssh user@bastion.aiops.baai.ac.cn -p 2224; touch /tmp/x",
    ],
)
def test_inventory_skips_unsafe_legacy_jump_commands(tmp_path, capsys, login):
    targets = fleet.load_inventory(_config(tmp_path, _record(login=login)))
    assert targets == ()
    assert "skipping fleet inventory line 1" in capsys.readouterr().err


def test_inventory_jump_user_override_ignores_replaced_login(tmp_path):
    config = _config(
        tmp_path,
        _record(login="invalid shell command"),
        jump_user="gupan",
    )
    assert _target(config).jump_login == "gupan@secure@10.1.2.3"


def test_inventory_bad_row_does_not_disable_other_devices(tmp_path, capsys):
    inventory = _record(login="invalid shell command") + _record(
        name="ascend",
        address="10.0.0.9",
        port=18306,
        login="operator@secure@10.0.0.9",
    )
    targets = fleet.load_inventory(_config(tmp_path, inventory))
    assert [target.name for target in targets] == ["ascend"]
    assert "skipping fleet inventory line 1" in capsys.readouterr().err


def test_inventory_reports_previously_silent_invalid_rows(tmp_path, capsys):
    inventory = _record(name="MUSA") + "musa|jump|missing\n" + _record(
        name="ascend",
        address="10.0.0.9",
        port=18306,
        login="operator@secure@10.0.0.9",
    )
    targets = fleet.load_inventory(_config(tmp_path, inventory))

    assert [target.name for target in targets] == ["ascend"]
    errors = capsys.readouterr().err
    assert "skipping fleet inventory line 1: invalid fleet device name" in errors
    assert "skipping fleet inventory line 2: expected 8 or 9 fields" in errors


def test_probe_command_uses_argv_and_quoted_container(tmp_path):
    config = _config(tmp_path, _record())
    command = fleet._probe_ssh_command(_target(config), config)
    assert command[0] == "ssh"
    assert "-T" in command
    assert command[-2] == config.bastion
    assert fleet._HOST_MARKER in command[-1]
    assert command[-1].endswith(
        "; exec sudo docker exec -i container_musa python3 -u -"
    )
    assert str(config.identity) in command


def test_start_proxy_only_passes_jump_user_in_jump_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    jump_config = _config(tmp_path, _record(), jump_user="gupan")
    jump_target = _target(jump_config)
    direct_inventory = "musa|direct|10.1.2.3|22|-|/workspace/kg|18303|-\n"
    direct_config = _config(tmp_path, direct_inventory, jump_user="gupan")
    direct_target = _target(direct_config)
    commands = []

    class FakeProcess:
        pid = 456
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        fleet.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command) or FakeProcess(),
    )
    monkeypatch.setattr(fleet, "proxy_process_matches", lambda *_args: True)

    assert fleet._start_proxy(
        jump_target, remote_port=18403, local_port=18100, config=jump_config
    ) == 456
    assert commands[-1][-2:] == ["--jump-user", "gupan"]

    assert fleet._start_proxy(
        direct_target, remote_port=18403, local_port=18101, config=direct_config
    ) == 456
    assert "--jump-user" not in commands[-1]


def test_port_probe_uses_reusable_socket(monkeypatch):
    calls = []

    class FakeSocket:
        def setsockopt(self, *args):
            calls.append(("setsockopt", args))

        def bind(self, address):
            calls.append(("bind", address))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(fleet.socket, "socket", lambda *_args: FakeSocket())
    assert fleet._port_available(18100) is True
    assert calls[0] == (
        "setsockopt",
        (fleet.socket.SOL_SOCKET, fleet.socket.SO_REUSEADDR, 1),
    )
    assert ("bind", ("127.0.0.1", 18100)) in calls


def test_probe_keeps_missing_port_unknown():
    output = fleet._JSON_MARKER + json.dumps(
        {
            "processes": [
                {
                    "pid": 321,
                    "port": None,
                    "backend": "musa",
                    "max_workers": 4,
                    "reachable": False,
                    "status": "unknown",
                }
            ]
        }
    )
    process = fleet._parse_probe_output(output)[0]
    assert process.port is None
    assert process.max_workers == 4
    assert process.status == "unknown"
    assert process.target_device is None


@pytest.mark.parametrize(
    ("status_payload", "expected"),
    [
        ({"status": "ok"}, None),
        ({"status": "ok", "target": None}, None),
        ({"status": "ok", "target": {}}, None),
        ({"status": "ok", "target": {"other": "CUDA"}}, None),
        ({"status": "ok", "target": {"device": ""}}, None),
        ({"status": "ok", "target": {"device": "   "}}, None),
        ({"status": "ok", "target": {"device": 123}}, None),
        (
            {
                "status": "ok",
                "capabilities": {
                    "candidate_admission": {
                        "version": 1,
                        "policy_sha256": "a" * 64,
                        "stages": ["preflight", 1],
                    }
                },
            },
            None,
        ),
        ({"status": "ok", "target": {"device": "  CUDA  "}}, "CUDA"),
    ],
)
def test_remote_probe_defensively_reads_target_device(
    monkeypatch, capsys, status_payload, expected
):
    cmdline = b"\0".join(
        [
            b"python3",
            b"-m",
            b"kernelgen_server.server",
            b"--port",
            b"18403",
            b"--backend",
            b"musa",
        ]
    ) + b"\0"

    class FakeOpener:
        def open(self, _url, timeout):
            assert timeout == 2
            return io.BytesIO(json.dumps(status_payload).encode("utf-8"))

    monkeypatch.setattr("glob.glob", lambda _pattern: ["/proc/321/cmdline"])
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: io.BytesIO(cmdline))
    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_args, **_kwargs: FakeOpener()
    )

    exec(fleet._REMOTE_PROBE, {})

    process = fleet._parse_probe_output(capsys.readouterr().out)[0]
    assert process.target_device == expected
    if status_payload.get("capabilities"):
        assert process.candidate_admission is False


def test_probe_target_uses_constant_stdin_and_hard_timeout(tmp_path, monkeypatch):
    config = _config(tmp_path, _record(), probe_timeout=7.0)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        payload = {
            "processes": [
                {
                    "pid": 9,
                    "port": 18403,
                    "backend": "musa",
                    "max_workers": 2,
                    "reachable": True,
                    "status": "ok",
                    "server_version": "v6.0",
                    "api_version": "v6.0",
                }
            ]
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=fleet._JSON_MARKER + json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(fleet.subprocess, "run", fake_run)
    result = fleet.probe_target(_target(config), config)
    assert result.state == "ok"
    assert captured["input"] == fleet._REMOTE_PROBE
    assert captured["timeout"] == 7.0
    assert "shell" not in captured


def test_probe_reachable_failure_is_degraded(tmp_path, monkeypatch):
    config = _config(tmp_path, _record())

    monkeypatch.setattr(
        fleet.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout=fleet._HOST_MARKER + "\n",
            stderr="private remote error",
        ),
    )

    result = fleet.probe_target(_target(config), config)
    assert result.state == "degraded"
    assert result.processes == ()


def test_probe_failure_is_sanitized_unknown(tmp_path, monkeypatch):
    config = _config(tmp_path, _record())

    def fail(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ssh", 5)

    monkeypatch.setattr(fleet.subprocess, "run", fail)
    result = fleet.probe_target(_target(config), config)
    assert result.state == "unknown"
    assert result.processes == ()


def test_reconcile_starts_and_reuses_proxy_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    config = _config(tmp_path, _record())
    target = _target(config)
    starts = []

    monkeypatch.setattr(fleet, "_port_available", lambda _port: True)
    monkeypatch.setattr(
        fleet,
        "_start_proxy",
        lambda target, **kwargs: starts.append((target.name, kwargs)) or 456,
    )
    monkeypatch.setattr(
        fleet,
        "proxy_process_matches",
        lambda pid, local, remote: pid == 456 and local == 18100 and remote == 18403,
    )

    first = fleet.reconcile(config, [_probe(target, target_device="MUSA")])
    assert first["devices"][0]["address"] == "10.1.2.3"
    assert first["devices"][0]["connection_state"] == "reachable"
    assert first["devices"][0]["kgs_state"] == "running"
    instance = first["instances"][0]
    assert instance["instance_id"] == "musa-18403"
    assert instance["target_hardware"] == "MUSA"
    assert instance["local_port"] == 18100
    assert instance["eval_server"] == "http://127.0.0.1:18100"
    assert instance["proxy_state"] == "running"
    assert instance["selectable"] is True

    second = fleet.reconcile(
        config, [_probe(target, pid=789, target_device="MUSA")]
    )
    assert second["instances"][0]["local_port"] == 18100
    assert len(starts) == 1


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"api_version": "v6.1"}, "incompatible_api"),
        ({"backend": None}, "target_unavailable"),
        ({"status": "degraded"}, "endpoint_unhealthy"),
        ({"candidate_admission": False}, "candidate_admission_missing"),
        ({"evaluation_binding": False}, "evaluation_binding_missing"),
        ({"scheduler_healthy": False}, "scheduler_unhealthy"),
    ],
)
def test_reconcile_rejects_incompatible_instances(
    tmp_path,
    monkeypatch,
    overrides,
    expected_reason,
):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    config = _config(tmp_path, _record())
    target = _target(config)
    monkeypatch.setattr(fleet, "_port_available", lambda _port: True)
    monkeypatch.setattr(fleet, "_start_proxy", lambda *_args, **_kwargs: 456)
    monkeypatch.setattr(fleet, "proxy_process_matches", lambda *_args: True)

    registry = fleet.reconcile(config, [_probe(target, **overrides)])

    assert registry["instances"][0]["selectable"] is False
    public = fleet.load_public_registry()
    assert public["instances"][0]["selectable"] is False
    assert public["instances"][0]["unavailable_reason"] == expected_reason
    assert "candidate_admission" not in public["instances"][0]
    assert "evaluation_binding" not in public["instances"][0]


def test_endpoint_gate_requires_target_hardware():
    instance = {
        "proxy_state": "running",
        "status": "ok",
        "api_version": "v6.2",
        "backend": "musa",
        "target_hardware": None,
        "candidate_admission": True,
        "evaluation_binding": True,
        "scheduler_healthy": True,
    }

    assert fleet._endpoint_selectable(instance) is False
    assert fleet._endpoint_unavailable_reason(instance) == "target_unavailable"


def test_endpoint_gate_reports_stopped_proxy():
    instance = {
        "proxy_state": "stopped",
        "status": "ok",
        "api_version": "v6.2",
        "backend": "musa",
        "target_hardware": "S5000",
        "candidate_admission": True,
        "evaluation_binding": True,
        "scheduler_healthy": True,
    }

    assert fleet._endpoint_selectable(instance) is False
    assert fleet._endpoint_unavailable_reason(instance) == "proxy_unavailable"


def test_reconcile_uses_fleet_owned_configured_proxy_port_once(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    config = _config(tmp_path, _record())
    target = _target(config)
    managed = tmp_path / "state" / "servers" / "fleet-musa-18403" / "config.json"
    fleet.atomic_write_json(
        managed,
        {
            "target": "remote",
            "fleet_device": "musa",
            "proxy_owner": "fleet",
            "port": 18403,
            "listen_port": 18107,
        },
    )
    starts = []
    monkeypatch.setattr(fleet, "_port_available", lambda _port: True)
    monkeypatch.setattr(
        fleet,
        "_start_proxy",
        lambda _target, **kwargs: starts.append(kwargs) or 456,
    )
    monkeypatch.setattr(fleet, "proxy_process_matches", lambda *_args: True)

    first = fleet.reconcile(config, [_probe(target)])
    second = fleet.reconcile(config, [_probe(target, pid=789)])

    assert first["instances"][0]["local_port"] == 18107
    assert second["instances"][0]["local_port"] == 18107
    assert starts == [{"remote_port": 18403, "local_port": 18107, "config": config}]


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_retired_fleet_deployment_has_no_managed_proxy_binding(
    tmp_path, monkeypatch, status
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    managed = tmp_path / "state" / "servers" / "fleet-musa-18403" / "config.json"
    fleet.atomic_write_json(
        managed,
        {
            "fleet_device": "musa",
            "proxy_owner": "fleet",
            "fleet_deployment_status": status,
            "port": 18403,
            "listen_port": 18107,
        },
    )

    assert fleet._managed_proxy_bindings() == {}


def test_request_scan_signals_running_daemon(monkeypatch):
    calls = []
    monkeypatch.setattr(fleet, "daemon_status", lambda: (True, 321))
    monkeypatch.setattr(fleet.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    assert fleet.request_scan() is True
    assert calls == [(321, fleet.signal.SIGHUP)]


def test_reconcile_restarts_proxy_when_connection_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    first_config = _config(tmp_path, _record(), jump_user="gupan")
    first_target = _target(first_config)
    active = set()
    starts = []
    stopped = []
    pids = iter((456, 789))

    monkeypatch.setattr(fleet, "_port_available", lambda _port: True)

    def start_proxy(target, **kwargs):
        pid = next(pids)
        active.add(pid)
        starts.append((target.address, kwargs["local_port"]))
        return pid

    def stop_proxy(pid, local_port, remote_port):
        stopped.append((pid, local_port, remote_port))
        active.discard(pid)

    monkeypatch.setattr(fleet, "_start_proxy", start_proxy)
    monkeypatch.setattr(fleet, "_stop_proxy", stop_proxy)
    monkeypatch.setattr(
        fleet,
        "proxy_process_matches",
        lambda pid, _local, _remote: pid in active,
    )

    first = fleet.reconcile(first_config, [_probe(first_target)])
    first_fingerprint = first["instances"][0]["connection_fingerprint"]

    second_config = _config(
        tmp_path,
        _record(address="10.9.8.7", login="operator@secure@10.9.8.7"),
        jump_user="gupan",
    )
    second_target = _target(second_config)
    second = fleet.reconcile(second_config, [_probe(second_target)])
    instance = second["instances"][0]

    assert starts == [("10.1.2.3", 18100), ("10.9.8.7", 18100)]
    assert stopped == [(456, 18100, 18403)]
    assert instance["proxy_pid"] == 789
    assert instance["connection_fingerprint"] != first_fingerprint


def test_reconcile_removes_device_deleted_from_inventory(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    config = _config(tmp_path, _record())
    target = _target(config)
    stopped = []

    monkeypatch.setattr(fleet, "_port_available", lambda _port: True)
    monkeypatch.setattr(fleet, "_start_proxy", lambda *_args, **_kwargs: 456)
    monkeypatch.setattr(fleet, "proxy_process_matches", lambda *_args: True)
    monkeypatch.setattr(fleet, "_stop_proxy", lambda *args: stopped.append(args))

    fleet.reconcile(config, [_probe(target)])
    removed = fleet.reconcile(config, [])

    assert removed["instances"] == []
    assert stopped == [(456, 18100, 18403)]


def test_reconcile_retains_unknown_then_removes_confirmed_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    config = _config(tmp_path, _record(), missing_grace_scans=2)
    target = _target(config)
    stopped = []

    monkeypatch.setattr(fleet, "_port_available", lambda _port: True)
    monkeypatch.setattr(fleet, "_start_proxy", lambda *_args, **_kwargs: 456)
    monkeypatch.setattr(fleet, "proxy_process_matches", lambda *_args: True)
    monkeypatch.setattr(fleet, "_stop_proxy", lambda *args: stopped.append(args))
    fleet.reconcile(config, [_probe(target)])

    unknown = fleet.DeviceProbe(target=target, state="unknown", processes=())
    retained = fleet.reconcile(config, [unknown])
    assert retained["devices"][0]["connection_state"] == "unreachable"
    assert retained["devices"][0]["kgs_state"] == "unknown"
    assert retained["instances"][0]["status"] == "unknown"
    assert stopped == []

    down = fleet.DeviceProbe(target=target, state="down", processes=())
    missing = fleet.reconcile(config, [down])
    assert missing["devices"][0]["connection_state"] == "reachable"
    assert missing["devices"][0]["kgs_state"] == "not_running"
    assert missing["instances"][0]["status"] == "missing"
    removed = fleet.reconcile(config, [down])
    assert removed["instances"] == []
    assert stopped == [(456, 18100, 18403)]


def test_public_registry_strips_pid_and_internal_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    fleet.atomic_write_json(
        fleet.registry_path(),
        {
            "schema_version": "1.0",
            "updated_at": "2026-09-10T00:00:00Z",
            "daemon_pid": 111,
            "devices": [
                {
                    "name": "musa",
                    "address": "10.1.2.3",
                    "connection_state": "reachable",
                    "kgs_state": "not_running",
                    "jump_login": "operator@secure@10.1.2.3",
                    "container": "container_musa",
                    "deploy_base": "/workspace/kg",
                }
            ],
            "instances": [
                {
                    "instance_id": "musa-18403",
                    "device": "musa",
                    "remote_port": 18403,
                    "local_port": 18100,
                    "eval_server": "http://127.0.0.1:18100",
                    "status": "ok",
                    "max_workers": 2,
                    "proxy_pid": 222,
                    "missing_scans": 0,
                    "connection_fingerprint": "private",
                }
            ],
        },
    )
    monkeypatch.setattr(fleet, "daemon_status", lambda: (True, 111))
    monkeypatch.setattr(fleet, "proxy_process_matches", lambda *_args: True)
    public = fleet.load_public_registry()
    assert public["daemon_running"] is True
    assert "daemon_pid" not in public
    assert public["devices"][0]["address"] == "10.1.2.3"
    assert set(public["devices"][0]) == {
        "name",
        "address",
        "connection_state",
        "kgs_state",
    }
    assert public["instances"][0]["max_workers"] == 2
    assert "proxy_pid" not in public["instances"][0]
    assert "missing_scans" not in public["instances"][0]
    assert "connection_fingerprint" not in public["instances"][0]
