"""Tests for minimal remote KGS deployment coordination."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kernelgen.service import fleet_daemon as fleet
from kernelgen.service import kgs_manager as manager


def _fleet_config(tmp_path: Path) -> fleet.FleetConfig:
    identity = tmp_path / "id_ed25519"
    identity.write_text("test key", encoding="utf-8")
    inventory = tmp_path / "hosts.conf"
    inventory.write_text("", encoding="utf-8")
    return fleet.FleetConfig(
        inventory=inventory,
        identity=identity,
        bastion="bastion.example",
        jump_user="",
        interval=30.0,
        probe_timeout=5.0,
        max_parallel=1,
        local_port_start=18100,
        local_port_end=18120,
        missing_grace_scans=2,
    )


def _target() -> fleet.InventoryTarget:
    return fleet.InventoryTarget(
        name="ascend",
        mode="jump",
        address="10.0.0.9",
        ssh_port=2224,
        container="container_ascend",
        deploy_base="/workspace/kg",
        expected_port=18306,
        jump_login="operator@secure@10.0.0.9",
        target_hardware="Ascend910B",
        devices=("0", "1"),
    )


def _resolved(version: str) -> dict:
    return {
        "version": version,
        "release": version,
        "branch": version,
        "commit": "a" * 40,
        "repository": "git@example.invalid:kernelgen_server.git",
        "strict_release": False,
    }


def _prepare(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KG_FLEET_HOME", str(tmp_path / "fleet"))
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    with manager._ACTIVE_DEPLOYMENTS_LOCK:
        manager._ACTIVE_DEPLOYMENTS.clear()
    config = _fleet_config(tmp_path)
    monkeypatch.setattr(manager, "_target", lambda machine: (config, _target()))
    monkeypatch.setattr(fleet, "_port_available", lambda _port: True)
    monkeypatch.setattr(manager.server, "_resolve_kgs_version", _resolved)


def _ready_deployment(tmp_path, monkeypatch) -> dict:
    _prepare(tmp_path, monkeypatch)
    _, task = manager.reserve_deployment("ascend", "development")
    record = task["record"]
    manager._update_deployment(record["deployment_id"], status="ready")
    manager.atomic_write_json(
        manager.server._config_path(record["instance_name"]),
        {
            "target": "remote",
            "proxy_owner": "fleet",
            "fleet_device": record["machine"],
            "fleet_deployment_status": "active",
            "port": record["remote_port"],
            "listen_port": record["local_port"],
        },
    )
    return record


def test_reservations_pin_version_and_allocate_distinct_ports(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)

    first, first_task = manager.reserve_deployment("ascend", "development")
    second, second_task = manager.reserve_deployment("ascend", "development")

    assert first["version"] == "development"
    assert first["instance_id"] == "ascend-18310"
    assert second["instance_id"] == "ascend-18311"
    assert first_task["record"]["local_port"] == 18100
    assert second_task["record"]["local_port"] == 18101
    assert first_task["resolved"]["commit"] == "a" * 40
    assert first_task["remote_kgs_root"].endswith("/" + "a" * 40)
    assert first_task["install_flaggems"] is True
    assert first_task["record"]["devices"] == ["0", "1"]
    args = manager._start_args(first_task)
    assert args.devices == "0,1"
    assert args.max_workers == 2

    public = manager.list_deployments()
    serialized = str(public).lower()
    assert len(public["deployments"]) == 2
    for secret in (
        "commit",
        "repository",
        "ssh_command",
        "container",
        "remote_kgs_root",
        "remote_port",
        "local_port",
        "instance_name",
        "install_flaggems",
        "devices",
    ):
        assert secret not in serialized


def test_reservation_requires_configured_devices(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    config = _fleet_config(tmp_path)
    monkeypatch.setattr(
        manager,
        "_target",
        lambda _machine: (config, replace(_target(), devices=())),
    )

    with pytest.raises(ValueError, match="machine devices are not configured"):
        manager.reserve_deployment("ascend", "development")


def test_reservation_keeps_native_only_choice_private(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)

    public, task = manager.reserve_deployment(
        "ascend",
        "development",
        install_flaggems=False,
    )

    assert task["install_flaggems"] is False
    assert "install_flaggems" not in public
    assert "install_flaggems" not in manager.list_deployments()["deployments"][0]


def test_fleet_instance_installs_flaggems_before_start(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    _, task = manager.reserve_deployment("ascend", "development")
    events = []

    monkeypatch.setattr(
        manager.server,
        "_create_instance_config",
        lambda _args, _instance: events.append("create") or {},
    )
    monkeypatch.setattr(
        manager.server,
        "_save_config",
        lambda _instance, _config: events.append("save"),
    )
    monkeypatch.setattr(
        manager,
        "_install_compatible_flaggems",
        lambda _instance, _config: events.append("install_flaggems"),
    )
    monkeypatch.setattr(
        manager.server,
        "_command_start",
        lambda _args: events.append("start") or 0,
    )

    manager._configure_fleet_owned_instance(task)

    assert events == ["create", "save", "install_flaggems", "start"]
    deployment = manager.list_deployments()["deployments"][0]
    assert deployment["status"] == "installing_flaggems"


def test_native_only_fleet_instance_starts_without_flaggems(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    _, task = manager.reserve_deployment(
        "ascend",
        "development",
        install_flaggems=False,
    )
    events = []

    monkeypatch.setattr(
        manager.server,
        "_create_instance_config",
        lambda _args, _instance: events.append("create") or {},
    )
    monkeypatch.setattr(manager.server, "_save_config", lambda _instance, _config: None)
    monkeypatch.setattr(
        manager,
        "_install_compatible_flaggems",
        lambda _instance, _config: events.append("unexpected_install"),
    )
    monkeypatch.setattr(
        manager.server,
        "_command_start",
        lambda _args: events.append("start") or 0,
    )

    manager._configure_fleet_owned_instance(task)

    assert events == ["create", "start"]


def test_remote_flaggems_install_reuses_cli_without_local_kgs_checkout(
    tmp_path, monkeypatch
):
    _prepare(tmp_path, monkeypatch)
    instance = "fleet-ascend-18310"
    config = {"flaggems_root": None}
    events = []

    monkeypatch.setattr(manager, "read_json", lambda _path, default=None: config)
    monkeypatch.setattr(
        manager.server,
        "_configured_flaggems",
        lambda observed: events.append(("configured", observed)) or None,
    )
    monkeypatch.setattr(
        manager.server,
        "_prepare_checkout",
        lambda *_args, **_kwargs: events.append(("unexpected_local_checkout",)),
    )
    monkeypatch.setattr(
        manager.server,
        "_command_install_flaggems",
        lambda args: events.append(("install", args.name, args.revision)) or 0,
    )

    manager._install_compatible_flaggems(instance, config)

    assert events == [
        ("configured", config),
        ("install", instance, None),
    ]


def test_flaggems_install_reuses_existing_configuration(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    instance = "fleet-ascend-18310"
    config = {"flaggems_root": "/managed/FlagGems"}
    installed = []

    monkeypatch.setattr(manager, "read_json", lambda _path, default=None: config)
    monkeypatch.setattr(
        manager.server,
        "_configured_flaggems",
        lambda _config: {"root": "/managed/FlagGems"},
    )
    monkeypatch.setattr(
        manager.server,
        "_command_install_flaggems",
        lambda _args: installed.append(True),
    )

    manager._install_compatible_flaggems(instance, config)

    assert installed == []


def test_dynamic_machine_name_uses_validated_vendor_backend(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)

    _, task = manager.reserve_deployment("ascend-dev", "development")

    assert task["backend"] == "npu"


def test_explicit_machine_backend_mapping_supports_custom_names(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv("KG_FLEET_MACHINE_BACKENDS", '{"accelerator-lab": "npu"}')

    _, task = manager.reserve_deployment("accelerator-lab", "development")

    assert task["backend"] == "npu"


def test_list_is_read_only_and_next_reservation_recovers_stale_deployment(
    tmp_path, monkeypatch
):
    _prepare(tmp_path, monkeypatch)
    stale_time = (
        datetime.now(timezone.utc) - timedelta(seconds=manager._STALE_DEPLOYMENT_SECONDS + 1)
    ).isoformat().replace("+00:00", "Z")
    manager.atomic_write_json(
        manager._deployments_path(),
        {
            "schema_version": "1.0",
            "deployments": [
                {
                    "deployment_id": "fleet-ascend-18310",
                    "instance_id": "ascend-18310",
                    "instance_name": "fleet-ascend-18310",
                    "machine": "ascend",
                    "version": "development",
                    "remote_port": 18310,
                    "local_port": 18100,
                    "status": "waiting_for_fleet",
                    "error_category": None,
                    "created_at": stale_time,
                    "updated_at": stale_time,
                }
            ],
        },
    )
    stopped = []
    monkeypatch.setattr(
        manager,
        "_stop_failed_instance",
        lambda instance_name: stopped.append(instance_name),
    )

    listed = manager.list_deployments()["deployments"][0]
    assert listed["status"] == "waiting_for_fleet"
    assert stopped == []

    replacement, _ = manager.reserve_deployment("ascend", "development")
    recovered = next(
        item
        for item in manager.list_deployments()["deployments"]
        if item["deployment_id"] == "fleet-ascend-18310"
    )

    assert recovered["status"] == "failed"
    assert recovered["error_category"] == "startup_interrupted"
    assert stopped == ["fleet-ascend-18310"]
    assert replacement["instance_id"] == "ascend-18310"


def test_reservation_recovers_orphaned_flaggems_install(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    stale_time = (
        datetime.now(timezone.utc)
        - timedelta(seconds=manager._STALE_DEPLOYMENT_SECONDS + 1)
    ).isoformat().replace("+00:00", "Z")
    manager.atomic_write_json(
        manager._deployments_path(),
        {
            "schema_version": "1.0",
            "deployments": [
                {
                    "deployment_id": "fleet-ascend-18310",
                    "instance_id": "ascend-18310",
                    "instance_name": "fleet-ascend-18310",
                    "machine": "ascend",
                    "version": "development",
                    "remote_port": 18310,
                    "local_port": 18100,
                    "status": "installing_flaggems",
                    "error_category": None,
                    "created_at": stale_time,
                    "updated_at": stale_time,
                }
            ],
        },
    )
    monkeypatch.setattr(manager, "_stop_failed_instance", lambda _instance: None)

    replacement, _ = manager.reserve_deployment("ascend", "development")
    recovered = next(
        item
        for item in manager.list_deployments()["deployments"]
        if item["deployment_id"] == "fleet-ascend-18310"
    )

    assert recovered["status"] == "failed"
    assert recovered["error_category"] == "flaggems_install_failed"
    assert replacement["instance_id"] == "ascend-18310"


def test_active_stale_deployment_is_not_recovered_or_stopped(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    stale_time = (
        datetime.now(timezone.utc) - timedelta(seconds=manager._STALE_DEPLOYMENT_SECONDS + 1)
    ).isoformat().replace("+00:00", "Z")
    deployment_id = "fleet-ascend-18310-active"
    manager.atomic_write_json(
        manager._deployments_path(),
        {
            "schema_version": "1.0",
            "deployments": [
                {
                    "deployment_id": deployment_id,
                    "instance_id": "ascend-18310",
                    "instance_name": "fleet-ascend-18310",
                    "machine": "ascend",
                    "version": "development",
                    "remote_port": 18310,
                    "local_port": 18100,
                    "status": "installing_flaggems",
                    "error_category": None,
                    "created_at": stale_time,
                    "updated_at": stale_time,
                }
            ],
        },
    )
    stopped = []
    monkeypatch.setattr(
        manager,
        "_stop_failed_instance",
        lambda instance: stopped.append(instance),
    )
    with manager._ACTIVE_DEPLOYMENTS_LOCK:
        manager._ACTIVE_DEPLOYMENTS.add(deployment_id)
    try:
        manager._recover_stale_deployments()
        replacement, _ = manager.reserve_deployment("ascend", "development")
    finally:
        with manager._ACTIVE_DEPLOYMENTS_LOCK:
            manager._ACTIVE_DEPLOYMENTS.discard(deployment_id)

    active = next(
        item
        for item in manager.list_deployments()["deployments"]
        if item["deployment_id"] == deployment_id
    )
    assert active["status"] == "installing_flaggems"
    assert stopped == []
    assert replacement["instance_id"] == "ascend-18311"


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_retired_fleet_config_does_not_reserve_ports(tmp_path, monkeypatch, status):
    _prepare(tmp_path, monkeypatch)
    manager.atomic_write_json(
        tmp_path / "state" / "servers" / "fleet-ascend-18310" / "config.json",
        {
            "fleet_device": "ascend",
            "proxy_owner": "fleet",
            "fleet_deployment_status": status,
            "port": 18310,
            "listen_port": 18100,
        },
    )

    public, _ = manager.reserve_deployment("ascend", "development")

    assert public["instance_id"] == "ascend-18310"


def test_failed_instance_config_is_retired_from_fleet_management(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    instance = "fleet-ascend-18310"
    manager.atomic_write_json(
        manager.server._config_path(instance),
        {
            "proxy_owner": "fleet",
            "fleet_device": "ascend",
            "fleet_deployment_status": "active",
            "port": 18310,
            "listen_port": 18100,
        },
    )
    monkeypatch.setattr(manager.server, "_command_stop", lambda _args: 0)
    monkeypatch.setattr(fleet, "request_scan", lambda: True)

    manager._stop_failed_instance(instance)

    config = manager.read_json(manager.server._config_path(instance))
    assert config["fleet_deployment_status"] == "failed"


def test_failed_instance_cleanup_continues_after_stop_key_error(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    instance = "fleet-ascend-18310"
    manager.atomic_write_json(
        manager.server._config_path(instance),
        {
            "proxy_owner": "fleet",
            "fleet_device": "ascend",
            "fleet_deployment_status": "active",
            "port": 18310,
            "listen_port": 18100,
        },
    )
    scans = []

    def fail_stop(_args):
        raise KeyError("target")

    monkeypatch.setattr(manager.server, "_command_stop", fail_stop)
    monkeypatch.setattr(fleet, "request_scan", lambda: scans.append(True) or True)

    manager._stop_failed_instance(instance)

    config = manager.read_json(manager.server._config_path(instance))
    assert config["fleet_deployment_status"] == "failed"
    assert scans == [True]


def test_stop_ready_managed_deployment_is_idempotent_and_releases_ports(
    tmp_path, monkeypatch
):
    record = _ready_deployment(tmp_path, monkeypatch)
    stopped = []
    scans = []
    monkeypatch.setattr(
        manager.server,
        "_command_stop",
        lambda args: stopped.append((args.name, args.timeout)) or 0,
    )
    monkeypatch.setattr(fleet, "request_scan", lambda: scans.append(True) or True)

    result = manager.stop_deployment(record["deployment_id"])
    repeated = manager.stop_deployment(record["deployment_id"])
    monkeypatch.setattr(
        fleet,
        "load_registry",
        lambda: {
            "instances": [
                {
                    "instance_id": record["instance_id"],
                    "device": record["machine"],
                    "remote_port": record["remote_port"],
                    "local_port": record["local_port"],
                }
            ]
        },
    )
    replacement, _ = manager.reserve_deployment("ascend", "development")

    assert result["status"] == "stopped"
    assert repeated["status"] == "stopped"
    assert stopped == [(record["instance_name"], 10.0)]
    assert scans == [True]
    config = manager.read_json(manager.server._config_path(record["instance_name"]))
    assert config["fleet_deployment_status"] == "stopped"
    assert replacement["instance_id"] == record["instance_id"]
    deployments = manager.list_deployments()["deployments"]
    assert len(deployments) == 2
    assert {item["status"] for item in deployments} == {"stopped", "starting"}
    assert "instance_name" not in str(deployments)


def test_stop_rejects_unmanaged_or_mismatched_configuration(tmp_path, monkeypatch):
    record = _ready_deployment(tmp_path, monkeypatch)
    config = manager.read_json(manager.server._config_path(record["instance_name"]))
    config["proxy_owner"] = "manual"
    manager.atomic_write_json(manager.server._config_path(record["instance_name"]), config)
    monkeypatch.setattr(
        manager.server,
        "_command_stop",
        lambda _args: pytest.fail("unmanaged instance must not be stopped"),
    )

    with pytest.raises(KeyError):
        manager.stop_deployment(record["deployment_id"])

    deployment = manager.list_deployments()["deployments"][0]
    assert deployment["status"] == "ready"


def test_stop_rejects_in_progress_deployment(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    public, _ = manager.reserve_deployment("ascend", "development")

    with pytest.raises(ValueError, match="not ready"):
        manager.stop_deployment(public["deployment_id"])


def test_stop_does_not_repeat_stopping_record(tmp_path, monkeypatch):
    record = _ready_deployment(tmp_path, monkeypatch)
    manager._update_deployment(record["deployment_id"], status="stopping")
    monkeypatch.setattr(
        manager.server,
        "_command_stop",
        lambda _args: pytest.fail("stopping instance must not be stopped again"),
    )

    result = manager.stop_deployment(record["deployment_id"])

    assert result["status"] == "stopping"


def test_stop_finalizes_stopping_record_when_config_is_already_stopped(
    tmp_path, monkeypatch
):
    record = _ready_deployment(tmp_path, monkeypatch)
    manager._update_deployment(record["deployment_id"], status="stopping")
    config = manager.read_json(manager.server._config_path(record["instance_name"]))
    config["fleet_deployment_status"] = "stopped"
    manager.atomic_write_json(manager.server._config_path(record["instance_name"]), config)
    monkeypatch.setattr(
        manager.server,
        "_command_stop",
        lambda _args: pytest.fail("already stopped instance must not be stopped again"),
    )
    monkeypatch.setattr(fleet, "request_scan", lambda: True)

    result = manager.stop_deployment(record["deployment_id"])

    assert result["status"] == "stopped"


def test_stop_failure_is_safe_and_keeps_ports_reserved(tmp_path, monkeypatch):
    record = _ready_deployment(tmp_path, monkeypatch)

    def fail(_args):
        raise RuntimeError("private remote stop detail")

    monkeypatch.setattr(manager.server, "_command_stop", fail)

    with pytest.raises(RuntimeError, match="managed KGS stop failed") as error:
        manager.stop_deployment(record["deployment_id"])

    assert "private remote stop detail" not in str(error.value)
    deployment = manager.list_deployments()["deployments"][0]
    assert deployment["status"] == "ready"
    assert deployment["error_category"] == "stop_failed"
    replacement, _ = manager.reserve_deployment("ascend", "development")
    assert replacement["instance_id"] == "ascend-18311"


def test_stop_bookkeeping_failure_stays_stopping_and_keeps_ports_reserved(
    tmp_path, monkeypatch
):
    record = _ready_deployment(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(
        manager.server,
        "_command_stop",
        lambda args: stopped.append(args.name) or 0,
    )

    def fail_save(_instance, _config):
        raise OSError("private config write detail")

    monkeypatch.setattr(manager.server, "_save_config", fail_save)

    with pytest.raises(RuntimeError, match="managed KGS stop failed") as error:
        manager.stop_deployment(record["deployment_id"])

    assert "private config write detail" not in str(error.value)
    assert stopped == [record["instance_name"]]
    deployment = manager.list_deployments()["deployments"][0]
    assert deployment["status"] == "stopping"
    assert deployment["error_category"] == "stop_failed"
    replacement, _ = manager.reserve_deployment("ascend", "development")
    assert replacement["instance_id"] == "ascend-18311"


def test_scheduler_initialization_waits_until_endpoint_is_ready(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    _, task = manager.reserve_deployment("ascend", "development")
    base_instance = {
        "instance_id": task["record"]["instance_id"],
        "proxy_pid": 123,
        "local_port": task["record"]["local_port"],
        "remote_port": task["record"]["remote_port"],
        "status": "ok",
        "api_version": "v6.2",
        "backend": "npu",
        "target_hardware": "Ascend910B",
        "candidate_admission": True,
        "evaluation_binding": True,
    }
    snapshots = [
        {"instances": [{**base_instance, "scheduler_healthy": False}]},
        {"instances": [{**base_instance, "scheduler_healthy": True}]},
    ]
    monkeypatch.setattr(manager, "_configure_fleet_owned_instance", lambda _task: None)
    monkeypatch.setattr(fleet, "load_registry", lambda: snapshots.pop(0))
    monkeypatch.setattr(fleet, "_proxy_state", lambda _instance: "running")
    monkeypatch.setattr(manager.time, "sleep", lambda _seconds: None)
    stopped = []
    monkeypatch.setattr(
        manager,
        "_stop_failed_instance",
        lambda instance_name: stopped.append(instance_name),
    )

    manager.deploy(task)

    deployment = manager.list_deployments()["deployments"][0]
    assert deployment["status"] == "ready"
    assert deployment["error_category"] is None
    assert task["record"]["deployment_id"] not in manager._active_deployments()
    assert stopped == []


def test_flaggems_install_failure_reports_safe_category_and_retires_instance(
    tmp_path, monkeypatch
):
    _prepare(tmp_path, monkeypatch)
    _, task = manager.reserve_deployment("ascend", "development")
    stopped = []

    def fail(_task):
        raise manager._FlagGemsInstallError("private remote install detail")

    monkeypatch.setattr(manager, "_configure_fleet_owned_instance", fail)
    monkeypatch.setattr(
        manager,
        "_stop_failed_instance",
        lambda instance_name: stopped.append(instance_name),
    )

    manager.deploy(task)

    public = manager.list_deployments()
    deployment = public["deployments"][0]
    assert deployment["status"] == "failed"
    assert deployment["error_category"] == "flaggems_install_failed"
    assert stopped == [task["record"]["instance_name"]]
    assert "private remote install detail" not in str(public)


def test_failed_deployment_reports_safe_category_and_stops_instance(
    tmp_path, monkeypatch
):
    _prepare(tmp_path, monkeypatch)
    _, task = manager.reserve_deployment("ascend", "development")
    stopped = []
    monkeypatch.setattr(manager, "_configure_fleet_owned_instance", lambda _task: None)
    monkeypatch.setattr(
        fleet,
        "load_registry",
        lambda: {
            "instances": [
                {
                    "instance_id": task["record"]["instance_id"],
                    "proxy_pid": 123,
                    "local_port": task["record"]["local_port"],
                    "remote_port": task["record"]["remote_port"],
                    "status": "ok",
                    "api_version": "v6.2",
                    "backend": "npu",
                    "target_hardware": "Ascend910B",
                    "candidate_admission": False,
                    "evaluation_binding": True,
                    "scheduler_healthy": True,
                }
            ]
        },
    )
    monkeypatch.setattr(fleet, "_proxy_state", lambda _instance: "running")
    monkeypatch.setattr(
        manager,
        "_stop_failed_instance",
        lambda instance_name: stopped.append(instance_name),
    )

    manager.deploy(task)

    deployment = manager.list_deployments()["deployments"][0]
    assert deployment["status"] == "failed"
    assert deployment["error_category"] == "candidate_admission_missing"
    assert stopped == [task["record"]["instance_name"]]


def test_deployment_failure_does_not_expose_exception_details(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    _, task = manager.reserve_deployment("ascend", "development")
    stopped = []

    def fail(_task):
        raise RuntimeError("private SSH failure detail")

    monkeypatch.setattr(manager, "_configure_fleet_owned_instance", fail)
    monkeypatch.setattr(
        manager,
        "_stop_failed_instance",
        lambda instance: stopped.append(instance),
    )

    manager.deploy(task)

    public = manager.list_deployments()
    deployment = public["deployments"][0]
    assert deployment["status"] == "failed"
    assert deployment["error_category"] == "deployment_failed"
    assert stopped == [task["record"]["instance_name"]]
    assert "private SSH failure detail" not in str(public)
