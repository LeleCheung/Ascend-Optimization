"""Minimal remote KGS version selection and Fleet-owned deployment."""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
import uuid
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from kernelgen.cli import server
from kernelgen.cli.state import atomic_write_json, cli_home, file_lock, read_json
from kernelgen.service import fleet_daemon as fleet


_BACKENDS = {
    "nvidia": "cuda",
    "tianshu": "iluvatar",
    "hygon": "hygon",
    "musa": "musa",
    "metax": "metax",
    "kunlun": "kunlunxin",
    "ascend": "npu",
    "ppu": "thead",
    "suiyuan": "enflame",
}
_PUBLIC_DEPLOYMENT_FIELDS = {
    "deployment_id",
    "instance_id",
    "machine",
    "version",
    "status",
    "error_category",
    "created_at",
    "updated_at",
}
_STALE_DEPLOYMENT_SECONDS = 300
_IN_PROGRESS_DEPLOYMENT_STATUSES = {
    "starting",
    "installing_flaggems",
    "waiting_for_fleet",
}
_RESERVED_DEPLOYMENT_STATUSES = _IN_PROGRESS_DEPLOYMENT_STATUSES | {"ready"}
_RETIRED_FLEET_STATUSES = {"failed", "stopped"}
_ALLOWED_BACKENDS = frozenset(_BACKENDS.values())
_ACTIVE_DEPLOYMENTS: set[str] = set()
_ACTIVE_DEPLOYMENTS_LOCK = threading.Lock()


class _FlagGemsInstallError(RuntimeError):
    pass


def _backend_for_machine(machine: str) -> str:
    raw = os.environ.get("KG_FLEET_MACHINE_BACKENDS", "").strip()
    if raw:
        try:
            configured = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid KG_FLEET_MACHINE_BACKENDS") from exc
        if not isinstance(configured, dict):
            raise RuntimeError("invalid KG_FLEET_MACHINE_BACKENDS")
        value = configured.get(machine)
        if value is not None:
            if not isinstance(value, str) or value not in _ALLOWED_BACKENDS:
                raise RuntimeError("invalid configured Fleet backend")
            return value
    exact = _BACKENDS.get(machine)
    if exact is not None:
        return exact
    for vendor, backend in _BACKENDS.items():
        if machine.startswith(f"{vendor}-") or machine.startswith(f"{vendor}_"):
            return backend
    raise ValueError("machine backend is not configured")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deployments_path() -> Path:
    return fleet.fleet_dir() / "deployments.json"


def _deployments_lock_path() -> Path:
    return fleet.fleet_dir() / "deployments.lock"


def _empty_deployments() -> dict:
    return {"schema_version": "1.0", "deployments": []}


def _load_deployments() -> dict:
    try:
        value = read_json(_deployments_path(), default=None)
    except (OSError, json.JSONDecodeError):
        return _empty_deployments()
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        return _empty_deployments()
    if not isinstance(value.get("deployments"), list):
        return _empty_deployments()
    return value


def _deployment_is_stale(record: dict, now: datetime) -> bool:
    if record.get("status") not in _IN_PROGRESS_DEPLOYMENT_STATUSES:
        return False
    updated_at = record.get("updated_at")
    if not isinstance(updated_at, str):
        return True
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return (now - updated.astimezone(timezone.utc)).total_seconds() > (
        _STALE_DEPLOYMENT_SECONDS
    )


def _public_deployment(record: dict) -> dict:
    return {
        key: record.get(key)
        for key in _PUBLIC_DEPLOYMENT_FIELDS
        if key in record
    }


def _active_deployments() -> set[str]:
    with _ACTIVE_DEPLOYMENTS_LOCK:
        return set(_ACTIVE_DEPLOYMENTS)


def _recover_stale_deployments() -> None:
    stale_instances = []
    active = _active_deployments()
    with file_lock(_deployments_lock_path()):
        value = _load_deployments()
        now = datetime.now(timezone.utc)
        changed = False
        deployments = []
        for item in value["deployments"]:
            if not isinstance(item, dict):
                continue
            if (
                item.get("deployment_id") not in active
                and _deployment_is_stale(item, now)
            ):
                error_category = (
                    "flaggems_install_failed"
                    if item.get("status") == "installing_flaggems"
                    else "startup_interrupted"
                )
                item = {
                    **item,
                    "status": "failed",
                    "error_category": error_category,
                    "updated_at": _now(),
                }
                instance_name = item.get("instance_name")
                if isinstance(instance_name, str):
                    stale_instances.append(instance_name)
                changed = True
            deployments.append(item)
        if changed:
            atomic_write_json(
                _deployments_path(),
                {"schema_version": "1.0", "deployments": deployments[-100:]},
            )
    for instance_name in stale_instances:
        _stop_failed_instance(instance_name)


def list_deployments() -> dict:
    value = _load_deployments()
    return {
        "schema_version": "1.0",
        "deployments": [
            _public_deployment(item)
            for item in value["deployments"]
            if isinstance(item, dict)
        ],
    }


def list_versions() -> dict:
    return {"schema_version": "1.0", "versions": server.list_kgs_versions()}


def _update_deployment(deployment_id: str, **changes) -> None:
    with file_lock(_deployments_lock_path()):
        value = _load_deployments()
        deployments = []
        found = False
        for item in value["deployments"]:
            if not isinstance(item, dict):
                continue
            if item.get("deployment_id") == deployment_id:
                item = {**item, **changes, "updated_at": _now()}
                found = True
            deployments.append(item)
        if found:
            atomic_write_json(
                _deployments_path(),
                {"schema_version": "1.0", "deployments": deployments[-100:]},
            )


def _target(machine: str) -> tuple[fleet.FleetConfig, fleet.InventoryTarget]:
    config = fleet.FleetConfig.from_env()
    config.validate()
    for target in fleet.load_inventory(config):
        if target.name == machine:
            return config, target
    raise KeyError(machine)


def _reserved_deployment_ports(machine: str, field: str) -> set[int]:
    return {
        item[field]
        for item in _load_deployments()["deployments"]
        if isinstance(item, dict)
        and item.get("machine") == machine
        and item.get("status") in _RESERVED_DEPLOYMENT_STATUSES
        and isinstance(item.get(field), int)
    }


def _retired_instance_ids() -> set[str]:
    latest: dict[str, str] = {}
    for item in _load_deployments()["deployments"]:
        if not isinstance(item, dict):
            continue
        instance_id = item.get("instance_id")
        status = item.get("status")
        if isinstance(instance_id, str) and isinstance(status, str):
            latest[instance_id] = status
    return {
        instance_id
        for instance_id, status in latest.items()
        if status in _RETIRED_FLEET_STATUSES
    }


def _used_remote_ports(machine: str) -> set[int]:
    retired = _retired_instance_ids()
    ports = _reserved_deployment_ports(machine, "remote_port")
    ports.update(
        {
            item["remote_port"]
            for item in fleet.load_registry().get("instances", [])
            if isinstance(item, dict)
            and item.get("device") == machine
            and item.get("instance_id") not in retired
            and isinstance(item.get("remote_port"), int)
        }
    )
    root = cli_home() / "servers"
    if root.is_dir():
        for path in root.iterdir():
            try:
                config = read_json(path / "config.json", default=None)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(config, dict)
                and config.get("fleet_device") == machine
                and config.get("fleet_deployment_status") not in _RETIRED_FLEET_STATUSES
                and isinstance(config.get("port"), int)
            ):
                ports.add(config["port"])
    return ports


def _used_local_ports() -> set[int]:
    retired = _retired_instance_ids()
    ports = {
        item["local_port"]
        for item in _load_deployments()["deployments"]
        if isinstance(item, dict)
        and item.get("status") in _RESERVED_DEPLOYMENT_STATUSES
        and isinstance(item.get("local_port"), int)
    }
    ports.update(
        {
            item["local_port"]
            for item in fleet.load_registry().get("instances", [])
            if isinstance(item, dict)
            and item.get("instance_id") not in retired
            and isinstance(item.get("local_port"), int)
        }
    )
    root = cli_home() / "servers"
    if root.is_dir():
        for path in root.iterdir():
            try:
                config = read_json(path / "config.json", default=None)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(config, dict)
                and config.get("fleet_deployment_status") not in _RETIRED_FLEET_STATUSES
                and isinstance(config.get("listen_port"), int)
            ):
                ports.add(config["listen_port"])
    return ports


def _allocate_remote_port(machine: str) -> int:
    try:
        start = int(os.environ.get("KG_FLEET_REMOTE_PORT_START", "18310"))
        end = int(os.environ.get("KG_FLEET_REMOTE_PORT_END", "18999"))
    except ValueError as exc:
        raise ValueError("invalid Fleet remote port range") from exc
    if not 1 <= start <= end <= 65535:
        raise ValueError("invalid Fleet remote port range")
    used = _used_remote_ports(machine)
    for port in range(start, end + 1):
        if port not in used:
            return port
    raise RuntimeError("no remote KGS ports are available")


def _base_ssh_command(target: fleet.InventoryTarget, config: fleet.FleetConfig) -> str:
    command = [
        "ssh",
        "-p",
        str(target.ssh_port),
        "-i",
        str(config.identity),
    ]
    if target.mode == "jump":
        command.extend(["-l", target.jump_login, config.bastion])
    else:
        command.append(f"root@{target.address}")
    return shlex.join(command)


def reserve_deployment(
    machine: str,
    version: str,
    *,
    install_flaggems: bool = True,
) -> tuple[dict, dict]:
    machine = machine.strip()
    version = version.strip()
    if not machine or not version:
        raise ValueError("machine and version are required")
    _recover_stale_deployments()
    config, target = _target(machine)
    if not target.devices:
        raise ValueError("machine devices are not configured")
    backend = _backend_for_machine(machine)
    resolved = server._resolve_kgs_version(version)
    with file_lock(_deployments_lock_path()):
        remote_port = _allocate_remote_port(machine)
        local_port = fleet._allocate_local_port(config, _used_local_ports())
        instance_name = f"fleet-{machine}-{remote_port}"
        deployment_id = f"{instance_name}-{uuid.uuid4().hex[:12]}"
        now = _now()
        record = {
            "deployment_id": deployment_id,
            "instance_id": f"{machine}-{remote_port}",
            "instance_name": instance_name,
            "machine": machine,
            "version": str(resolved["version"]),
            "devices": list(target.devices),
            "remote_port": remote_port,
            "local_port": local_port,
            "status": "starting",
            "error_category": None,
            "created_at": now,
            "updated_at": now,
        }
        value = _load_deployments()
        deployments = [
            item
            for item in value["deployments"]
            if isinstance(item, dict)
            and item.get("deployment_id") != deployment_id
        ]
        deployments.append(record)
        atomic_write_json(
            _deployments_path(),
            {"schema_version": "1.0", "deployments": deployments[-100:]},
        )
    task = {
        "record": record,
        "resolved": resolved,
        "backend": backend,
        "ssh_command": _base_ssh_command(target, config),
        "container": target.container,
        "remote_kgs_root": f"{target.deploy_base}/kgs/{resolved['commit']}",
        "remote_state_root": f"{target.deploy_base}/servers/{instance_name}",
        "remote_env_file": f"{target.deploy_base}/deployment.env.sh",
        "install_flaggems": install_flaggems,
    }
    return (
        _public_deployment(record),
        task,
    )


def _start_args(task: dict) -> Namespace:
    record = task["record"]
    return Namespace(
        name=record["instance_name"],
        target="remote",
        backend=task["backend"],
        devices=",".join(record["devices"]),
        timing="auto",
        port=None,
        remote_port=record["remote_port"],
        listen_port=record["local_port"],
        max_workers=len(record["devices"]),
        max_streams=3,
        ssh_command=task["ssh_command"],
        container=task["container"],
        remote_python="python3",
        remote_kgs_root=task["remote_kgs_root"],
        remote_state_root=task["remote_state_root"],
        remote_env_file=task["remote_env_file"],
        python=sys.executable,
        kgs_root=None,
        kgs_version=record["version"],
        resolved_kgs=task["resolved"],
        startup_timeout=120.0,
    )


def _install_compatible_flaggems(instance: str, config: dict) -> None:
    configured = server._configured_flaggems(
        read_json(server._config_path(instance), default=config)
    )
    if configured is not None:
        return
    server._command_install_flaggems(Namespace(name=instance, revision=None))


def _configure_fleet_owned_instance(task: dict) -> None:
    record = task["record"]
    instance = record["instance_name"]
    args = _start_args(task)
    with file_lock(server._server_root(instance) / "lifecycle.lock"):
        existing = read_json(server._config_path(instance), default=None)
        if (
            existing is None
            or existing.get("fleet_deployment_status") in _RETIRED_FLEET_STATUSES
        ):
            with file_lock(server._server_locks_root() / "deployment.lock"):
                config = server._create_instance_config(args, instance)
            config["fleet_device"] = record["machine"]
            config["proxy_owner"] = "fleet"
            config["fleet_deployment_status"] = "active"
            server._save_config(instance, config)
        else:
            config = existing
    if task["install_flaggems"]:
        _update_deployment(record["deployment_id"], status="installing_flaggems")
        try:
            _install_compatible_flaggems(instance, config)
        except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
            raise _FlagGemsInstallError from exc
    server._command_start(args)


def _readiness_category(instance: dict) -> str | None:
    if instance.get("api_version") != fleet._required_protocol():
        return "api_version_mismatch"
    if instance.get("candidate_admission") is not True:
        return "candidate_admission_missing"
    if instance.get("evaluation_binding") is not True:
        return "evaluation_binding_missing"
    if instance.get("scheduler_healthy") is not True:
        return "scheduler_unhealthy"
    if not instance.get("backend") or not instance.get("target_hardware"):
        return "target_unavailable"
    return None


def _stop_failed_instance(instance_name: str) -> None:
    try:
        server._command_stop(Namespace(name=instance_name, timeout=10.0))
    except (KeyError, OSError, RuntimeError, ValueError):
        pass
    try:
        config = read_json(server._config_path(instance_name), default=None)
        if isinstance(config, dict) and config.get("proxy_owner") == "fleet":
            config["fleet_deployment_status"] = "failed"
            server._save_config(instance_name, config)
    except (OSError, RuntimeError, ValueError):
        pass
    try:
        fleet.request_scan()
    except (OSError, RuntimeError, ValueError):
        pass


def _deployment_config_matches(record: dict, config: object) -> bool:
    return bool(
        isinstance(config, dict)
        and config.get("proxy_owner") == "fleet"
        and config.get("fleet_device") == record.get("machine")
        and config.get("port") == record.get("remote_port")
        and config.get("listen_port") == record.get("local_port")
    )


def stop_deployment(deployment_id: str) -> dict:
    finalized = None
    with file_lock(_deployments_lock_path()):
        value = _load_deployments()
        deployments = [
            item for item in value["deployments"] if isinstance(item, dict)
        ]
        record = next(
            (
                item
                for item in deployments
                if item.get("deployment_id") == deployment_id
            ),
            None,
        )
        if record is None:
            raise KeyError(deployment_id)
        status = record.get("status")
        if status == "stopped":
            return _public_deployment(record)
        if status not in {"ready", "stopping"}:
            raise ValueError("deployment is not ready to stop")
        instance_name = record.get("instance_name")
        if not isinstance(instance_name, str):
            raise KeyError(deployment_id)
        try:
            config = read_json(server._config_path(instance_name), default=None)
        except (OSError, ValueError) as exc:
            raise RuntimeError("managed KGS configuration is unavailable") from exc
        if not _deployment_config_matches(record, config):
            raise KeyError(deployment_id)
        config_status = config.get("fleet_deployment_status")
        if status == "stopping":
            if config_status == "stopped":
                record = {
                    **record,
                    "status": "stopped",
                    "error_category": None,
                    "updated_at": _now(),
                }
                deployments = [
                    record if item.get("deployment_id") == deployment_id else item
                    for item in deployments
                ]
                atomic_write_json(
                    _deployments_path(),
                    {"schema_version": "1.0", "deployments": deployments[-100:]},
                )
                finalized = _public_deployment(record)
            else:
                return _public_deployment(record)
        else:
            if config_status in _RETIRED_FLEET_STATUSES:
                raise KeyError(deployment_id)
            record = {
                **record,
                "status": "stopping",
                "error_category": None,
                "updated_at": _now(),
            }
            deployments = [
                record if item.get("deployment_id") == deployment_id else item
                for item in deployments
            ]
            atomic_write_json(
                _deployments_path(),
                {"schema_version": "1.0", "deployments": deployments[-100:]},
            )

    if finalized is not None:
        try:
            fleet.request_scan()
        except (OSError, RuntimeError, ValueError):
            pass
        return finalized

    try:
        server._command_stop(Namespace(name=instance_name, timeout=10.0))
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        _update_deployment(
            deployment_id,
            status="ready",
            error_category="stop_failed",
        )
        raise RuntimeError("managed KGS stop failed") from exc

    try:
        config = read_json(server._config_path(instance_name), default=None)
        if not _deployment_config_matches(record, config):
            raise RuntimeError("managed KGS configuration changed during stop")
        config["fleet_deployment_status"] = "stopped"
        server._save_config(instance_name, config)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        _update_deployment(
            deployment_id,
            status="stopping",
            error_category="stop_failed",
        )
        raise RuntimeError("managed KGS stop failed") from exc

    _update_deployment(deployment_id, status="stopped", error_category=None)
    try:
        fleet.request_scan()
    except (OSError, RuntimeError, ValueError):
        pass
    return next(
        item
        for item in list_deployments()["deployments"]
        if item.get("deployment_id") == deployment_id
    )


def _deploy(task: dict) -> None:
    record = task["record"]
    deployment_id = record["deployment_id"]
    try:
        _configure_fleet_owned_instance(task)
    except _FlagGemsInstallError:
        _update_deployment(
            deployment_id,
            status="failed",
            error_category="flaggems_install_failed",
        )
        _stop_failed_instance(record["instance_name"])
        return
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError):
        _update_deployment(
            deployment_id,
            status="failed",
            error_category="deployment_failed",
        )
        _stop_failed_instance(record["instance_name"])
        return

    _update_deployment(deployment_id, status="waiting_for_fleet")
    deadline = time.monotonic() + 150
    last_category = None
    permanent_categories = {
        "api_version_mismatch",
        "candidate_admission_missing",
        "evaluation_binding_missing",
    }
    while time.monotonic() < deadline:
        registry = fleet.load_registry()
        instance = next(
            (
                item
                for item in registry.get("instances", [])
                if isinstance(item, dict)
                and item.get("instance_id") == record["instance_id"]
            ),
            None,
        )
        if instance is not None and fleet._proxy_state(instance) == "running":
            category = _readiness_category(instance)
            if category is None and fleet._endpoint_selectable(
                {**instance, "proxy_state": "running"}
            ):
                _update_deployment(
                    deployment_id,
                    status="ready",
                    error_category=None,
                )
                return
            if category is not None:
                last_category = category
            if (
                instance.get("status") == "ok"
                and category in permanent_categories
            ):
                _update_deployment(
                    deployment_id,
                    status="failed",
                    error_category=category,
                )
                _stop_failed_instance(record["instance_name"])
                return
        time.sleep(1)

    _update_deployment(
        deployment_id,
        status="failed",
        error_category=last_category or "startup_timeout",
    )
    _stop_failed_instance(record["instance_name"])


def deploy(task: dict) -> None:
    deployment_id = task["record"]["deployment_id"]
    with _ACTIVE_DEPLOYMENTS_LOCK:
        _ACTIVE_DEPLOYMENTS.add(deployment_id)
    try:
        _deploy(task)
    finally:
        with _ACTIVE_DEPLOYMENTS_LOCK:
            _ACTIVE_DEPLOYMENTS.discard(deployment_id)
