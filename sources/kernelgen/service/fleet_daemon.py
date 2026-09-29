"""Discover remote KGS endpoints and keep their fixed-port SSH proxies alive.

The daemon deliberately reuses the existing stdio HTTP proxy unchanged: every
routable KGS endpoint gets one local listener and one persistent SSH process.
FastAPI only reads the atomically written registry and never opens SSH itself.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from kernelgen.cli.state import atomic_write_json, read_json
from kernelgen.service.hints import DEVICE_PRESETS


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INVENTORY = _REPO_ROOT / "tests" / "hosts.md"
_PROXY_SCRIPT = _REPO_ROOT / "scripts" / "remote_server" / "run_persistent_remote_http_proxy.sh"
_JSON_MARKER = "__KG_FLEET_JSON__"
_HOST_MARKER = "__KG_FLEET_HOST_OK__"
_NAME_RE = re.compile(r"[a-z0-9_-]+")
_ADDRESS_RE = re.compile(r"[A-Za-z0-9._:-]+")
_CONTAINER_RE = re.compile(r"[A-Za-z0-9_.-]+")
_LOGIN_RE = re.compile(r"[A-Za-z0-9@#._:-]+")
_USER_RE = re.compile(r"[A-Za-z0-9._-]+")
_DEVICE_TOKEN_RE = re.compile(r"[A-Za-z0-9._:-]+")
_PROXY_STARTUP_TIMEOUT = 3.0
_PROXY_STARTUP_POLL_INTERVAL = 0.05
_PROXY_TERMINATE_GRACE = 1.0
_PROXY_SHUTDOWN_TIMEOUT = 3.0
_PROXY_SHUTDOWN_POLL_INTERVAL = 0.1
_RETIRED_FLEET_DEPLOYMENT_STATUSES = {"failed", "stopped"}


_REMOTE_PROBE = r'''
import glob
import json
import os
import urllib.request


def option_value(argv, name):
    prefix = name + "="
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def is_kgs(argv):
    for index, value in enumerate(argv):
        if value == "-m" and index + 1 < len(argv):
            if argv[index + 1] == "kernelgen_server.server":
                return True
        if value == "kernelgen_server.server":
            return True
        if os.path.basename(value) == "kernelgen-server":
            return True
    return False


opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
processes = []
for cmdline_path in glob.glob("/proc/[0-9]*/cmdline"):
    try:
        pid = int(cmdline_path.split("/")[2])
        raw = open(cmdline_path, "rb").read()
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    except (OSError, ValueError):
        continue
    if not argv or not is_kgs(argv):
        continue

    port_raw = option_value(argv, "--port")
    backend = option_value(argv, "--backend")
    max_workers_raw = option_value(argv, "--max-workers")
    try:
        port = int(port_raw) if port_raw is not None else None
    except ValueError:
        port = None
    if port is not None and not 1 <= port <= 65535:
        port = None
    try:
        max_workers = int(max_workers_raw) if max_workers_raw is not None else None
    except ValueError:
        max_workers = None
    if max_workers is not None and max_workers <= 0:
        max_workers = None

    record = {
        "pid": pid,
        "port": port,
        "backend": backend,
        "max_workers": max_workers,
        "reachable": False,
        "status": "unknown",
        "server_version": None,
        "api_version": None,
        "target_device": None,
        "candidate_admission": False,
        "evaluation_binding": False,
        "scheduler_healthy": False,
    }
    if port is not None:
        try:
            with opener.open("http://127.0.0.1:%d/status" % port, timeout=2) as response:
                status = json.load(response)
            target = status.get("target")
            target_device = None
            if isinstance(target, dict):
                value = target.get("device")
                if isinstance(value, str) and value.strip():
                    target_device = value.strip()
            capabilities = status.get("capabilities")
            capabilities = capabilities if isinstance(capabilities, dict) else {}
            admission = capabilities.get("candidate_admission")
            digest = admission.get("policy_sha256") if isinstance(admission, dict) else None
            candidate_admission = bool(
                isinstance(admission, dict)
                and type(admission.get("version")) is int
                and admission.get("version") == 1
                and isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
                and isinstance(admission.get("stages"), list)
                and all(
                    isinstance(stage, str) for stage in admission.get("stages")
                )
                and "preflight" in admission.get("stages")
            )
            upload = capabilities.get("operator_bundle_upload")
            evaluation_binding = bool(
                isinstance(upload, dict)
                and upload.get("enabled") is True
                and upload.get("evaluation_binding") is True
            )
            scheduler = status.get("scheduler")
            scheduler_healthy = bool(
                isinstance(scheduler, dict)
                and isinstance(scheduler.get("healthy"), int)
                and scheduler.get("healthy") > 0
                and scheduler.get("checking") == 0
                and scheduler.get("broken") == 0
                and isinstance(scheduler.get("max_active"), int)
                and scheduler.get("max_active") > 0
            )
            record.update({
                "reachable": True,
                "status": status.get("status", "unknown"),
                "backend": status.get("backend", backend),
                "server_version": status.get("server_version"),
                "api_version": status.get("api_version"),
                "target_device": target_device,
                "candidate_admission": candidate_admission,
                "evaluation_binding": evaluation_binding,
                "scheduler_healthy": scheduler_healthy,
            })
        except Exception:
            pass
    processes.append(record)

print("__KG_FLEET_JSON__" + json.dumps({"processes": processes}, separators=(",", ":")))
'''.strip()


@dataclass(frozen=True)
class FleetConfig:
    inventory: Path
    identity: Path
    bastion: str
    jump_user: str
    interval: float
    probe_timeout: float
    max_parallel: int
    local_port_start: int
    local_port_end: int
    missing_grace_scans: int

    @classmethod
    def from_env(cls) -> "FleetConfig":
        return cls(
            inventory=Path(
                os.environ.get("MULTI_DEVICE_INVENTORY", str(_DEFAULT_INVENTORY))
            ).expanduser().resolve(),
            identity=Path(
                os.environ.get("MULTI_DEVICE_IDENTITY", "~/.ssh/id_ed25519")
            ).expanduser().resolve(),
            bastion=os.environ.get(
                "MULTI_DEVICE_BASTION", "bastion.aiops.baai.ac.cn"
            ),
            jump_user=os.environ.get("KG_JUMP_USER", "").strip(),
            interval=float(os.environ.get("KG_FLEET_SCAN_INTERVAL", "60")),
            probe_timeout=float(os.environ.get("KG_FLEET_PROBE_TIMEOUT", "15")),
            max_parallel=int(os.environ.get("KG_FLEET_MAX_PARALLEL", "3")),
            local_port_start=int(os.environ.get("KG_FLEET_LOCAL_PORT_START", "18100")),
            local_port_end=int(os.environ.get("KG_FLEET_LOCAL_PORT_END", "18999")),
            missing_grace_scans=int(
                os.environ.get("KG_FLEET_MISSING_GRACE_SCANS", "3")
            ),
        )

    def validate(self) -> None:
        if not self.inventory.is_file():
            raise ValueError("fleet inventory is not readable")
        if not self.identity.is_file():
            raise ValueError("fleet SSH identity is not readable")
        if not _ADDRESS_RE.fullmatch(self.bastion):
            raise ValueError("invalid fleet bastion")
        if self.jump_user and not _USER_RE.fullmatch(self.jump_user):
            raise ValueError("invalid KG_JUMP_USER")
        if self.interval <= 0 or self.probe_timeout <= 0:
            raise ValueError("fleet intervals must be positive")
        if self.max_parallel <= 0:
            raise ValueError("KG_FLEET_MAX_PARALLEL must be positive")
        if not 1 <= self.local_port_start <= self.local_port_end <= 65535:
            raise ValueError("invalid fleet local port range")
        if self.missing_grace_scans <= 0:
            raise ValueError("KG_FLEET_MISSING_GRACE_SCANS must be positive")


@dataclass(frozen=True)
class InventoryTarget:
    name: str
    mode: str
    address: str
    ssh_port: int
    container: str
    deploy_base: str
    expected_port: int
    jump_login: str
    target_hardware: str
    devices: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeProcess:
    pid: int
    port: int | None
    backend: str | None
    max_workers: int | None
    status: str
    server_version: str | None
    api_version: str | None
    target_device: str | None = None
    candidate_admission: bool = False
    evaluation_binding: bool = False
    scheduler_healthy: bool = False


@dataclass(frozen=True)
class DeviceProbe:
    target: InventoryTarget
    state: str
    processes: tuple[ProbeProcess, ...]


def fleet_dir() -> Path:
    override = os.environ.get("KG_FLEET_HOME")
    if override:
        return Path(override).expanduser().resolve()
    cli_override = os.environ.get("KERNELGEN_CLI_HOME")
    if cli_override:
        return Path(cli_override).expanduser().resolve() / "fleet"
    return (Path.home() / ".kernelgen" / "fleet").resolve()


def registry_path() -> Path:
    return fleet_dir() / "registry.json"


def daemon_pid_path() -> Path:
    return fleet_dir() / "daemon.pid"


def daemon_log_path() -> Path:
    return fleet_dir() / "daemon.log"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded_string(value: object, *, maximum: int = 80) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:maximum]


def _port(value: str, field: str) -> int:
    if not value.isdigit() or not 1 <= int(value) <= 65535:
        raise ValueError(f"invalid {field}")
    return int(value)


def _devices(value: str, device: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate device token for {device}")
    if any(not _DEVICE_TOKEN_RE.fullmatch(item) for item in devices):
        raise ValueError(f"invalid device token for {device}")
    return devices


def _normalize_jump_login(
    value: str, *, bastion: str, ssh_port: int, device: str
) -> str:
    parts = shlex.split(value)
    if parts and parts[0] == "ssh":
        if len(parts) != 4 or parts[2] != "-p":
            raise ValueError(f"invalid jump command for {device}")
        if _port(parts[3], "jump command port") != ssh_port:
            raise ValueError(f"jump command port mismatch for {device}")
        suffix = f"@{bastion}"
        if not parts[1].endswith(suffix):
            raise ValueError(f"jump command bastion mismatch for {device}")
        value = parts[1][: -len(suffix)]
    if not _LOGIN_RE.fullmatch(value):
        raise ValueError(f"invalid jump login for {device}")
    return value


def load_inventory(config: FleetConfig) -> tuple[InventoryTarget, ...]:
    presets = {item.name: item.target_hardware for item in DEVICE_PRESETS}
    targets: list[InventoryTarget] = []
    seen: set[str] = set()
    lines = config.inventory.read_text(encoding="utf-8").splitlines()
    if config.inventory.suffix == ".md":
        starts = [i for i, line in enumerate(lines) if line == "```kg-hosts"]
        if len(starts) != 1:
            raise ValueError("hosts.md must contain exactly one kg-hosts code block")
        start = starts[0] + 1
        try:
            end = lines.index("```", start)
        except ValueError as exc:
            raise ValueError("unterminated kg-hosts code block") from exc
        # Keep original line numbers in diagnostics.
        lines = [""] * start + lines[start:end]
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) not in (8, 9):
            print(
                f"skipping fleet inventory line {line_number}: expected 8 or 9 fields",
                file=sys.stderr,
                flush=True,
            )
            continue
        name, mode, address, ssh_port_raw, container, deploy_base = fields[:6]
        expected_port_raw = fields[6]
        devices_raw = "" if len(fields) == 8 else fields[7]
        jump_value = fields[7] if len(fields) == 8 else fields[8]
        try:
            if name in seen:
                raise ValueError(f"duplicate fleet device: {name}")
            if not _NAME_RE.fullmatch(name):
                raise ValueError("invalid fleet device name")
            if mode not in ("jump", "direct"):
                raise ValueError(f"invalid fleet mode for {name}")
            if not _ADDRESS_RE.fullmatch(address):
                raise ValueError(f"invalid fleet address for {name}")
            ssh_port = _port(ssh_port_raw, "SSH port")
            expected_port = _port(expected_port_raw, "expected KGS port")
            devices = _devices(devices_raw, name)
            if not deploy_base.startswith("/"):
                raise ValueError(f"invalid deployment path for {name}")
            if mode == "jump":
                if not _CONTAINER_RE.fullmatch(container):
                    raise ValueError(f"invalid container for {name}")
                jump_login = (
                    f"{config.jump_user}@secure@{address}"
                    if config.jump_user
                    else _normalize_jump_login(
                        jump_value,
                        bastion=config.bastion,
                        ssh_port=ssh_port,
                        device=name,
                    )
                )
            else:
                if container != "-" or jump_value != "-":
                    raise ValueError(f"invalid direct-mode record for {name}")
                jump_login = "-"
        except ValueError as exc:
            print(
                f"skipping fleet inventory line {line_number}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        targets.append(
            InventoryTarget(
                name=name,
                mode=mode,
                address=address,
                ssh_port=ssh_port,
                container=container,
                deploy_base=deploy_base,
                expected_port=expected_port,
                jump_login=jump_login,
                target_hardware=presets.get(name, name),
                devices=devices,
            )
        )
        seen.add(name)
    return tuple(targets)


def _probe_ssh_command(target: InventoryTarget, config: FleetConfig) -> list[str]:
    remote = ["python3", "-u", "-"]
    if target.container != "-":
        remote = ["sudo", "docker", "exec", "-i", target.container, *remote]
    remote_command = (
        f"printf '%s\\n' {shlex.quote(_HOST_MARKER)}; exec {shlex.join(remote)}"
    )
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=2",
        "-o",
        "TCPKeepAlive=yes",
        "-p",
        str(target.ssh_port),
        "-i",
        str(config.identity),
    ]
    if target.mode == "jump":
        command.extend(
            ["-l", target.jump_login, config.bastion, remote_command]
        )
    else:
        command.extend([f"root@{target.address}", remote_command])
    return command


def _parse_probe_output(output: str) -> tuple[ProbeProcess, ...]:
    payload = None
    for line in output.splitlines():
        if line.startswith(_JSON_MARKER):
            payload = line[len(_JSON_MARKER) :]
    if payload is None:
        raise ValueError("fleet probe returned no result")
    value = json.loads(payload)
    raw_processes = value.get("processes")
    if not isinstance(raw_processes, list):
        raise ValueError("fleet probe returned invalid processes")
    processes: list[ProbeProcess] = []
    for raw in raw_processes:
        if not isinstance(raw, dict):
            raise ValueError("fleet probe returned an invalid process")
        pid = raw.get("pid")
        port = raw.get("port")
        max_workers = raw.get("max_workers")
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("fleet probe returned an invalid pid")
        if port is not None and (not isinstance(port, int) or not 1 <= port <= 65535):
            raise ValueError("fleet probe returned an invalid port")
        if max_workers is not None and (
            not isinstance(max_workers, int) or max_workers <= 0
        ):
            max_workers = None
        if port is None:
            status = "unknown"
        elif raw.get("reachable"):
            remote_status = (_bounded_string(raw.get("status")) or "unknown").lower()
            status = "ok" if remote_status == "ok" else "degraded"
        else:
            status = "unreachable"
        target_device = raw.get("target_device")
        if isinstance(target_device, str):
            target_device = target_device.strip()[:80] or None
        else:
            target_device = None
        processes.append(
            ProbeProcess(
                pid=pid,
                port=port,
                backend=_bounded_string(raw.get("backend")),
                max_workers=max_workers,
                status=status,
                server_version=_bounded_string(raw.get("server_version")),
                api_version=_bounded_string(raw.get("api_version")),
                target_device=target_device,
                candidate_admission=raw.get("candidate_admission") is True,
                evaluation_binding=raw.get("evaluation_binding") is True,
                scheduler_healthy=raw.get("scheduler_healthy") is True,
            )
        )
    return tuple(sorted(processes, key=lambda item: (item.port or 0, item.pid)))


def _host_marker_present(output: str | bytes | None) -> bool:
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return isinstance(output, str) and any(
        line.strip() == _HOST_MARKER for line in output.splitlines()
    )


def probe_target(target: InventoryTarget, config: FleetConfig) -> DeviceProbe:
    try:
        result = subprocess.run(
            _probe_ssh_command(target, config),
            input=_REMOTE_PROBE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.probe_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        state = "degraded" if _host_marker_present(exc.stdout) else "unknown"
        return DeviceProbe(target=target, state=state, processes=())
    except OSError:
        return DeviceProbe(target=target, state="unknown", processes=())
    host_reachable = _host_marker_present(result.stdout)
    if result.returncode != 0:
        state = "degraded" if host_reachable else "unknown"
        return DeviceProbe(target=target, state=state, processes=())
    try:
        processes = _parse_probe_output(result.stdout)
    except (ValueError, json.JSONDecodeError):
        state = "degraded" if host_reachable else "unknown"
        return DeviceProbe(target=target, state=state, processes=())
    if not processes:
        state = "down"
    elif any(process.status == "ok" for process in processes):
        state = "ok"
    else:
        state = "degraded"
    return DeviceProbe(target=target, state=state, processes=processes)


def scan_fleet(
    config: FleetConfig, targets: Iterable[InventoryTarget] | None = None
) -> tuple[DeviceProbe, ...]:
    selected = tuple(targets if targets is not None else load_inventory(config))
    if not selected:
        return ()
    workers = min(config.max_parallel, len(selected))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(probe_target, target, config): target for target in selected}
        probes = [future.result() for future in concurrent.futures.as_completed(futures)]
    return tuple(sorted(probes, key=lambda item: item.target.name))


def _endpoint_groups(probe: DeviceProbe) -> list[dict]:
    by_port: dict[int, list[ProbeProcess]] = {}
    for process in probe.processes:
        if process.port is not None:
            by_port.setdefault(process.port, []).append(process)
    endpoints = []
    rank = {"ok": 3, "degraded": 2, "unreachable": 1, "unknown": 0}
    for port, processes in sorted(by_port.items()):
        representative = max(processes, key=lambda item: rank[item.status])
        endpoints.append(
            {
                "remote_port": port,
                "process_count": len(processes),
                "backend": representative.backend,
                "max_workers": representative.max_workers,
                "status": representative.status,
                "server_version": representative.server_version,
                "api_version": representative.api_version,
                "target_device": representative.target_device,
                "candidate_admission": representative.candidate_admission,
                "evaluation_binding": representative.evaluation_binding,
                "scheduler_healthy": representative.scheduler_healthy,
            }
        )
    return endpoints


def _process_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _option_from_argv(argv: list[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def proxy_process_matches(pid: int, local_port: int, remote_port: int) -> bool:
    argv = _process_cmdline(pid)
    if not argv or not any(
        os.path.basename(value) == "ssh_stdio_http_mux_proxy.py" for value in argv
    ):
        return False
    return (
        _option_from_argv(argv, "--listen-port") == str(local_port)
        and _option_from_argv(argv, "--remote-port") == str(remote_port)
    )


def _required_protocol() -> str:
    from kernelgen.cli.server import _kgs_lock

    return str(_kgs_lock()["validated_protocol"])


def _endpoint_unavailable_reason(instance: dict) -> str | None:
    if instance.get("proxy_state") != "running":
        return "proxy_unavailable"
    if instance.get("status") != "ok":
        return "endpoint_unhealthy"
    if instance.get("api_version") != _required_protocol():
        return "incompatible_api"
    if not instance.get("backend") or not instance.get("target_hardware"):
        return "target_unavailable"
    if instance.get("candidate_admission") is not True:
        return "candidate_admission_missing"
    if instance.get("evaluation_binding") is not True:
        return "evaluation_binding_missing"
    if instance.get("scheduler_healthy") is not True:
        return "scheduler_unhealthy"
    return None


def _endpoint_selectable(instance: dict) -> bool:
    return _endpoint_unavailable_reason(instance) is None


def _port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _allocate_local_port(config: FleetConfig, used: set[int]) -> int:
    for port in range(config.local_port_start, config.local_port_end + 1):
        if port not in used and _port_available(port):
            return port
    raise RuntimeError("no local fleet proxy ports are available")


def _proxy_log_path(instance_id: str) -> Path:
    return fleet_dir() / "proxies" / f"{instance_id}.log"


def _start_proxy(
    target: InventoryTarget,
    *,
    remote_port: int,
    local_port: int,
    config: FleetConfig,
) -> int | None:
    log_path = _proxy_log_path(f"{target.name}-{remote_port}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(_PROXY_SCRIPT),
        "--device",
        target.name,
        "--listen-port",
        str(local_port),
        "--remote-port",
        str(remote_port),
        "--inventory",
        str(config.inventory),
        "--identity",
        str(config.identity),
    ]
    if config.jump_user and target.mode == "jump":
        command.extend(["--jump-user", config.jump_user])
    try:
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"{_now()} proxy start failed: {exc}\n")
        return None
    deadline = time.monotonic() + _PROXY_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"{_now()} proxy exited during startup with code "
                    f"{process.returncode}\n"
                )
            return None
        if proxy_process_matches(process.pid, local_port, remote_port):
            return process.pid
        time.sleep(_PROXY_STARTUP_POLL_INTERVAL)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"{_now()} proxy did not become ready within "
            f"{_PROXY_STARTUP_TIMEOUT:g} seconds\n"
        )
    process.terminate()
    try:
        process.wait(timeout=_PROXY_TERMINATE_GRACE)
    except subprocess.TimeoutExpired:
        process.kill()
    return None


def _stop_proxy(pid: int, local_port: int, remote_port: int) -> None:
    if not proxy_process_matches(pid, local_port, remote_port):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + _PROXY_SHUTDOWN_TIMEOUT
    while time.monotonic() < deadline:
        if not proxy_process_matches(pid, local_port, remote_port):
            return
        time.sleep(_PROXY_SHUTDOWN_POLL_INTERVAL)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def empty_registry() -> dict:
    return {
        "schema_version": "1.0",
        "updated_at": None,
        "daemon_pid": None,
        "devices": [],
        "instances": [],
    }


def load_registry() -> dict:
    try:
        value = read_json(registry_path(), default=None)
    except (OSError, json.JSONDecodeError):
        return empty_registry()
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        return empty_registry()
    if not isinstance(value.get("devices"), list) or not isinstance(
        value.get("instances"), list
    ):
        return empty_registry()
    return value


def load_public_registry() -> dict:
    """Return the browser-safe view of the daemon registry."""
    registry = load_registry()
    device_fields = {
        "name",
        "address",
        "target_hardware",
        "devices",
        "state",
        "connection_state",
        "kgs_state",
        "expected_port",
        "expected_port_match",
        "process_count",
        "unknown_port_processes",
    }
    instance_fields = {
        "instance_id",
        "device",
        "target_hardware",
        "remote_port",
        "local_port",
        "eval_server",
        "backend",
        "max_workers",
        "status",
        "server_version",
        "api_version",
        "process_count",
        "expected_port",
        "last_seen_at",
        "proxy_state",
        "selectable",
    }
    running, _ = daemon_status()
    public_instances = []
    for item in registry.get("instances", []):
        if not isinstance(item, dict):
            continue
        public = {key: value for key, value in item.items() if key in instance_fields}
        public["proxy_state"] = _proxy_state(item)
        projected = {**item, "proxy_state": public["proxy_state"]}
        public["unavailable_reason"] = _endpoint_unavailable_reason(projected)
        public["selectable"] = public["unavailable_reason"] is None
        public_instances.append(public)
    return {
        "schema_version": "1.0",
        "updated_at": registry.get("updated_at"),
        "daemon_running": running,
        "devices": [
            {key: value for key, value in item.items() if key in device_fields}
            for item in registry.get("devices", [])
            if isinstance(item, dict)
        ],
        "instances": public_instances,
    }


def _previous_instances() -> dict[str, dict]:
    return {
        item["instance_id"]: item
        for item in load_registry().get("instances", [])
        if isinstance(item, dict) and isinstance(item.get("instance_id"), str)
    }


def _managed_proxy_bindings() -> dict[tuple[str, int], int]:
    from kernelgen.cli.state import cli_home

    root = cli_home() / "servers"
    if not root.is_dir():
        return {}
    bindings: dict[tuple[str, int], int] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            config = read_json(path / "config.json", default=None)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(config, dict)
            or config.get("proxy_owner") != "fleet"
            or config.get("fleet_deployment_status")
            in _RETIRED_FLEET_DEPLOYMENT_STATUSES
        ):
            continue
        device = config.get("fleet_device")
        remote_port = config.get("port")
        local_port = config.get("listen_port")
        if (
            isinstance(device, str)
            and isinstance(remote_port, int)
            and isinstance(local_port, int)
        ):
            bindings[(device, remote_port)] = local_port
    return bindings


def _proxy_state(instance: dict) -> str:
    pid = instance.get("proxy_pid")
    local_port = instance.get("local_port")
    remote_port = instance.get("remote_port")
    if not all(isinstance(value, int) for value in (pid, local_port, remote_port)):
        return "stopped"
    return (
        "running"
        if proxy_process_matches(pid, local_port, remote_port)
        else "stopped"
    )


def _connection_fingerprint(target: InventoryTarget, config: FleetConfig) -> str:
    payload = json.dumps(
        [
            target.mode,
            target.address,
            target.ssh_port,
            target.container,
            target.deploy_base,
            target.jump_login,
            str(config.identity),
            config.bastion,
            config.jump_user,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _machine_states(state: str) -> tuple[str, str]:
    if state == "ok":
        return ("reachable", "running")
    if state == "down":
        return ("reachable", "not_running")
    if state == "degraded":
        return ("reachable", "degraded")
    return ("unreachable", "unknown")


def reconcile(config: FleetConfig, probes: Iterable[DeviceProbe]) -> dict:
    previous = _previous_instances()
    managed_bindings = _managed_proxy_bindings()
    used_ports = {
        item["local_port"]
        for item in previous.values()
        if isinstance(item.get("local_port"), int)
    }
    used_ports.update(managed_bindings.values())
    now = _now()
    devices = []
    instances: list[dict] = []
    seen: set[str] = set()
    probe_by_name = {probe.target.name: probe for probe in probes}

    for name in sorted(probe_by_name):
        probe = probe_by_name[name]
        endpoints = _endpoint_groups(probe)
        connection_state, kgs_state = _machine_states(probe.state)
        devices.append(
            {
                "name": name,
                "address": probe.target.address,
                "target_hardware": probe.target.target_hardware,
                "devices": list(probe.target.devices),
                "state": probe.state,
                "connection_state": connection_state,
                "kgs_state": kgs_state,
                "expected_port": probe.target.expected_port,
                "expected_port_match": (
                    None
                    if probe.state == "unknown"
                    else any(
                        endpoint["remote_port"] == probe.target.expected_port
                        for endpoint in endpoints
                    )
                ),
                "process_count": len(probe.processes),
                "unknown_port_processes": sum(
                    process.port is None for process in probe.processes
                ),
            }
        )
        for endpoint in endpoints:
            remote_port = endpoint["remote_port"]
            instance_id = f"{name}-{remote_port}"
            seen.add(instance_id)
            old = previous.get(instance_id, {})
            local_port = old.get("local_port")
            old_proxy_running = _proxy_state(old) == "running"
            managed_local_port = managed_bindings.get((name, remote_port))
            if (
                managed_local_port is not None
                and local_port != managed_local_port
                and old_proxy_running
            ):
                old_pid = old.get("proxy_pid")
                if all(
                    isinstance(value, int)
                    for value in (old_pid, local_port, remote_port)
                ):
                    _stop_proxy(old_pid, local_port, remote_port)
                old = {**old, "proxy_pid": None}
                old_proxy_running = False
            connection_fingerprint = _connection_fingerprint(probe.target, config)
            if old and old.get("connection_fingerprint") != connection_fingerprint:
                old_pid = old.get("proxy_pid")
                if old_proxy_running and all(
                    isinstance(value, int)
                    for value in (old_pid, local_port, remote_port)
                ):
                    _stop_proxy(old_pid, local_port, remote_port)
                old = {**old, "proxy_pid": None}
                old_proxy_running = False
            if managed_local_port is not None:
                local_port = managed_local_port
                used_ports.add(local_port)
            elif not isinstance(local_port, int) or (
                not old_proxy_running and not _port_available(local_port)
            ):
                local_port = _allocate_local_port(config, used_ports)
                used_ports.add(local_port)
            candidate = {
                **old,
                **endpoint,
                "instance_id": instance_id,
                "device": name,
                "target_hardware": (
                    endpoint["target_device"] or probe.target.target_hardware
                ),
                "connection_fingerprint": connection_fingerprint,
                "expected_port": probe.target.expected_port,
                "local_port": local_port,
                "eval_server": f"http://127.0.0.1:{local_port}",
                "last_seen_at": now,
                "missing_scans": 0,
            }
            if _proxy_state(candidate) != "running":
                candidate["proxy_pid"] = _start_proxy(
                    probe.target,
                    remote_port=remote_port,
                    local_port=local_port,
                    config=config,
                )
            candidate["proxy_state"] = _proxy_state(candidate)
            candidate["selectable"] = _endpoint_selectable(candidate)
            instances.append(candidate)

    for instance_id, old in previous.items():
        if instance_id in seen:
            continue
        device = old.get("device")
        probe = probe_by_name.get(device)
        if probe is None:
            pid = old.get("proxy_pid")
            local_port = old.get("local_port")
            remote_port = old.get("remote_port")
            if all(isinstance(value, int) for value in (pid, local_port, remote_port)):
                _stop_proxy(pid, local_port, remote_port)
            continue
        if probe.state == "unknown":
            retained = dict(old)
            retained["status"] = "unknown"
            retained["proxy_state"] = _proxy_state(retained)
            retained["selectable"] = False
            instances.append(retained)
            continue
        missing_scans = int(old.get("missing_scans", 0)) + 1
        if missing_scans < config.missing_grace_scans:
            retained = dict(old)
            retained["status"] = "missing"
            retained["missing_scans"] = missing_scans
            retained["proxy_state"] = _proxy_state(retained)
            retained["selectable"] = False
            instances.append(retained)
            continue
        pid = old.get("proxy_pid")
        local_port = old.get("local_port")
        remote_port = old.get("remote_port")
        if all(isinstance(value, int) for value in (pid, local_port, remote_port)):
            _stop_proxy(pid, local_port, remote_port)

    registry = {
        "schema_version": "1.0",
        "updated_at": now,
        "daemon_pid": os.getpid(),
        "devices": devices,
        "instances": sorted(
            instances, key=lambda item: (item.get("device", ""), item.get("remote_port", 0))
        ),
    }
    atomic_write_json(registry_path(), registry)
    return registry


def run_once(config: FleetConfig) -> dict:
    config.validate()
    targets = load_inventory(config)
    return reconcile(config, scan_fleet(config, targets))


def _daemon_process_matches(pid: int) -> bool:
    argv = _process_cmdline(pid)
    return bool(
        argv
        and "kernelgen.service.fleet_daemon" in argv
        and "run" in argv
    )


def daemon_status() -> tuple[bool, int | None]:
    try:
        pid = int(daemon_pid_path().read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return False, None
    return (_daemon_process_matches(pid), pid)


def request_scan() -> bool:
    running, pid = daemon_status()
    if not running or pid is None:
        return False
    os.kill(pid, signal.SIGHUP)
    return True


def run_daemon(config: FleetConfig, *, once: bool = False) -> int:
    config.validate()
    state_dir = fleet_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "daemon.lock"
    stop_requested = False
    scan_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    def request_rescan(_signum, _frame) -> None:
        nonlocal scan_requested
        scan_requested = True

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Fleet daemon is already running", file=sys.stderr)
            return 1
        daemon_pid_path().write_text(f"{os.getpid()}\n", encoding="utf-8")
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGHUP, request_rescan)
        try:
            while not stop_requested:
                scan_requested = False
                try:
                    run_once(config)
                except Exception as exc:
                    print(f"fleet reconciliation failed: {exc}", file=sys.stderr, flush=True)
                if once:
                    break
                deadline = time.monotonic() + config.interval
                while (
                    not stop_requested
                    and not scan_requested
                    and time.monotonic() < deadline
                ):
                    time.sleep(min(0.5, deadline - time.monotonic()))
        finally:
            try:
                daemon_pid_path().unlink()
            except FileNotFoundError:
                pass
    return 0


def start_daemon(config: FleetConfig) -> int:
    running, pid = daemon_status()
    if running:
        print(f"Fleet daemon already running (pid {pid})")
        return 0
    fleet_dir().mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "kernelgen.service.fleet_daemon", "run"]
    with daemon_log_path().open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(_REPO_ROOT),
            env=os.environ.copy(),
        )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        running, pid = daemon_status()
        if running:
            print(f"Fleet daemon started (pid {pid})")
            print(f"Registry: {registry_path()}")
            print(f"Logs: {daemon_log_path()}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.1)
    print(f"Fleet daemon failed to start; check {daemon_log_path()}", file=sys.stderr)
    return 1


def stop_daemon() -> int:
    running, pid = daemon_status()
    if not running or pid is None:
        print("Fleet daemon is not running")
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _daemon_process_matches(pid):
            time.sleep(0.1)
        if _daemon_process_matches(pid):
            print("Fleet daemon did not stop", file=sys.stderr)
            return 1
        print("Fleet daemon stopped")
    for item in load_registry().get("instances", []):
        if not isinstance(item, dict):
            continue
        values = (item.get("proxy_pid"), item.get("local_port"), item.get("remote_port"))
        if all(isinstance(value, int) for value in values):
            _stop_proxy(*values)
    return 0


def _scan_rows(probes: Iterable[DeviceProbe]) -> list[dict]:
    rows = []
    for probe in probes:
        if not probe.processes:
            rows.append(
                {
                    "device": probe.target.name,
                    "expected_port": probe.target.expected_port,
                    "pid": None,
                    "port": None,
                    "backend": None,
                    "max_workers": None,
                    "server_version": None,
                    "api_version": None,
                    "status": probe.state,
                    "expected_match": False if probe.state != "unknown" else None,
                }
            )
            continue
        for process in probe.processes:
            rows.append(
                {
                    "device": probe.target.name,
                    "expected_port": probe.target.expected_port,
                    "pid": process.pid,
                    "port": process.port,
                    "backend": process.backend,
                    "max_workers": process.max_workers,
                    "server_version": process.server_version,
                    "api_version": process.api_version,
                    "status": process.status,
                    "expected_match": (
                        process.port == probe.target.expected_port
                        if process.port is not None
                        else False
                    ),
                }
            )
    return rows


def print_scan(probes: Iterable[DeviceProbe], *, as_json: bool = False) -> None:
    rows = _scan_rows(probes)
    if as_json:
        print(json.dumps({"schema_version": "1.0", "rows": rows}, indent=2))
        return
    print(
        f"{'DEVICE':<10} {'EXPECTED':<8} {'PID':<8} {'PORT':<7} "
        f"{'BACKEND':<12} {'WORKERS':<7} {'SERVER':<9} {'API':<8} "
        f"{'STATUS':<11} MATCH"
    )
    for row in rows:
        value = lambda item: "-" if item is None else str(item)
        match = "-" if row["expected_match"] is None else (
            "yes" if row["expected_match"] else "no"
        )
        print(
            f"{value(row['device']):<10} {value(row['expected_port']):<8} "
            f"{value(row['pid']):<8} {value(row['port']):<7} "
            f"{value(row['backend']):<12} {value(row['max_workers']):<7} "
            f"{value(row['server_version']):<9} "
            f"{value(row['api_version']):<8} {value(row['status']):<11} {match}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start", help="start the fleet daemon")
    subparsers.add_parser("stop", help="stop the daemon and managed proxies")
    subparsers.add_parser("status", help="show daemon and registry status")
    run_parser = subparsers.add_parser("run", help="run the daemon in foreground")
    run_parser.add_argument("--once", action="store_true")
    scan_parser = subparsers.add_parser("scan", help="scan remote KGS processes once")
    scan_parser.add_argument("--device", default="")
    scan_parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = FleetConfig.from_env()
    if args.command == "start":
        try:
            config.validate()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return start_daemon(config)
    if args.command == "stop":
        return stop_daemon()
    if args.command == "status":
        running, pid = daemon_status()
        registry = load_registry()
        print(f"daemon: {'running' if running else 'stopped'}" + (f" (pid {pid})" if pid else ""))
        print(f"updated_at: {registry.get('updated_at') or '-'}")
        print(f"instances: {len(registry.get('instances', []))}")
        return 0 if running else 1
    try:
        config.validate()
        if args.command == "run":
            return run_daemon(config, once=args.once)
        targets = load_inventory(config)
        selected = {name for name in args.device.split(",") if name}
        if selected:
            known = {target.name for target in targets}
            missing = selected - known
            if missing:
                raise ValueError(f"unknown fleet device: {', '.join(sorted(missing))}")
            targets = tuple(target for target in targets if target.name in selected)
        print_scan(scan_fleet(config, targets), as_json=args.json)
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
