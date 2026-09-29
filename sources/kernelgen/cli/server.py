"""Local KernelGen Server checkout, configuration, and process lifecycle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from contextlib import ExitStack
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import yaml

from kernelgen.cli.models import ServerProcessRecord
from kernelgen.cli.state import (
    atomic_write_json,
    cli_home,
    file_lock,
    process_is_alive,
    process_start_identity,
    read_json,
)
DEVICE_ENVIRONMENTS = {
    "enflame": ("TOPS_VISIBLE_DEVICES",),
    "cuda": ("CUDA_VISIBLE_DEVICES",),
    "iluvatar": ("CUDA_VISIBLE_DEVICES",),
    "hygon": ("HIP_VISIBLE_DEVICES",),
    "musa": ("MTHREADS_VISIBLE_DEVICES", "MUSA_VISIBLE_DEVICES"),
    "metax": ("CUDA_VISIBLE_DEVICES",),
    "kunlunxin": ("CUDA_VISIBLE_DEVICES",),
    "npu": ("ASCEND_RT_VISIBLE_DEVICES",),
    "thead": ("CUDA_VISIBLE_DEVICES",),
    "mlu": (),
}


RESULT_MARKER = "__KERNELGEN_REMOTE_SERVER_RESULT_V1__"


INSTANCE_NAME_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62})\Z")


def _validate_instance_name(value: str) -> str:
    if INSTANCE_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "server name must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-' (maximum 63 characters)"
        )
    return value


def _servers_root() -> Path:
    return cli_home() / "servers"


def _server_locks_root() -> Path:
    return cli_home() / "locks" / "servers"


def _server_root(instance: str) -> Path:
    return _servers_root() / _validate_instance_name(instance)


def _config_path(instance: str) -> Path:
    return _server_root(instance) / "config.json"


def _process_path(instance: str) -> Path:
    return _server_root(instance) / "process.json"


def _log_path(instance: str) -> Path:
    return _server_root(instance) / "kernelgen-server.log"


def _proxy_log_path(instance: str) -> Path:
    return _server_root(instance) / "remote-proxy.log"


def _lock_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "deployment" / "kgs.lock.yaml"


def _read_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"manifest not found: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def _kgs_lock() -> dict:
    lock = _read_yaml(_lock_manifest_path())
    kgs = lock.get("kgs")
    if not isinstance(kgs, dict):
        raise ValueError("deployment lock is missing KGS configuration")
    return lock


def _validate_kgs_branch(value: str) -> str:
    if not value or value.startswith("-") or any(
        character.isspace() or character == "\0" for character in value
    ):
        raise ValueError("KGS version must be a Git branch name")
    _run(["git", "check-ref-format", f"refs/heads/{value}"])
    return value


_GIT_REMOTE_TIMEOUT_SECONDS = 20.0


def _run_git_remote(command: list[str]) -> str:
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return _run(
        command,
        timeout=_GIT_REMOTE_TIMEOUT_SECONDS,
        env=environment,
    )


def list_kgs_versions() -> list[dict[str, object]]:
    """Return browser-safe KGS version labels, with the deployment lock first."""
    lock = _kgs_lock()
    kgs = lock["kgs"]
    repository = str(kgs["repository"])
    recommended = str(kgs["release"])
    try:
        rows = _run_git_remote(
            ["git", "ls-remote", "--heads", repository]
        ).splitlines()
    except (OSError, RuntimeError):
        rows = []
    branches: list[str] = []
    for row in rows:
        fields = row.split()
        if len(fields) != 2 or not fields[1].startswith("refs/heads/"):
            continue
        branch = fields[1][len("refs/heads/") :]
        if branch and branch != recommended:
            branches.append(branch)
    return [
        {"name": recommended, "recommended": True},
        *(
            {"name": branch, "recommended": False}
            for branch in sorted(set(branches))
        ),
    ]


def _resolve_kgs_version(version: str | None) -> dict[str, object]:
    """Resolve a user-facing version once while keeping the commit internal."""
    lock = _kgs_lock()
    kgs = lock["kgs"]
    recommended = str(kgs["release"])
    selected = (version or "").strip() or recommended
    if selected == recommended:
        return {
            "version": recommended,
            "release": recommended,
            "branch": None,
            "commit": str(kgs["commit"]),
            "repository": str(kgs["repository"]),
            "strict_release": True,
        }
    branch = _validate_kgs_branch(selected)
    reference = f"refs/heads/{branch}"
    rows = _run_git_remote(
        ["git", "ls-remote", "--exit-code", str(kgs["repository"]), reference]
    ).splitlines()
    matches = [
        fields
        for row in rows
        if len(fields := row.split()) == 2 and fields[1] == reference
    ]
    if len(matches) != 1 or not re.fullmatch(r"[0-9a-f]{40}", matches[0][0]):
        raise ValueError(f"cannot resolve KGS version: {selected}")
    return {
        "version": selected,
        "release": selected,
        "branch": branch,
        "commit": matches[0][0],
        "repository": str(kgs["repository"]),
        "strict_release": False,
    }


def _command_error_summary(stderr: str) -> str:
    # Keep actionable diagnostics, never arbitrary subprocess output or URLs.
    if "PRIVATE KEY" in stderr:
        return "diagnostic omitted: sensitive content"
    lines = [line.strip() for line in stderr.splitlines()
             if re.match(r"\s*(?:ERROR:|error:|ModuleNotFoundError:|ImportError:|.*BackendUnavailable:)", line)]
    detail = " | ".join(lines[-3:])
    detail = re.sub(r"https?://\S+", "[redacted URL]", detail)
    for name, value in os.environ.items():
        if value and re.search(r"token|password|secret|(?:^|_)key$", name, re.I):
            detail = detail.replace(value, "[redacted]")
    detail = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", detail)
    detail = re.sub(r'''(?i)((?:token|password|secret|api[_-]?key|authorization)\s*[:=]\s*)(?:"[^"]*"|'[^']*'|\S+)''', r"\1[redacted]", detail)
    return detail[:1000]


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"command timed out ({command[0]} {command[1]})"
        ) from exc
    if result.returncode != 0:
        detail = _command_error_summary(result.stderr or "")
        raise RuntimeError(
            f"command failed with exit code {result.returncode} "
            f"({command[0]} {command[1]})" + (f": {detail}" if detail else "")
        )
    return result.stdout.strip()


def _parse_ssh_command(value: str) -> list[str]:
    if any(character in value for character in ("\n", "\r", "\0")):
        raise ValueError("--ssh-command must be a single command line")
    try:
        command = shlex.split(value)
    except ValueError as exc:
        raise ValueError(f"invalid --ssh-command: {exc}") from exc
    if not command or Path(command[0]).name != "ssh":
        raise ValueError("--ssh-command must invoke ssh directly")
    if any(option == "-t" or option.startswith("-tt") for option in command[1:]):
        raise ValueError("--ssh-command must not allocate a TTY")
    return command


def _validate_container(value: str) -> str:
    if value != "-" and re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError("--container must be '-' or a Docker container name")
    return value


def _remote_exec_command(
    ssh_command: list[str],
    *,
    container: str,
    command: list[str],
) -> list[str]:
    remote = command
    if container != "-":
        remote = ["sudo", "docker", "exec", "-i", container, *command]
    managed_options = [
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=6",
        "-o",
        "ServerAliveCountMax=30",
        "-o",
        "TCPKeepAlive=yes",
    ]
    return [ssh_command[0], *managed_options, *ssh_command[1:], shlex.join(remote)]



def _remote_management_command(remote_python: str, action: str, payload: dict) -> list[str]:
    """KG bootstraps Git only; target-side lifecycle code comes from the pinned KGS."""
    encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    bootstrap = r"""
set -eu
kgs_root=$1
case "$kgs_root" in "~/"*) kgs_root="$HOME/${kgs_root#'~/'}";; esac
if [ ! -e "$kgs_root" ]; then
    (
        if [ -n "$5" ]; then
            set -a
            source "$5" >/dev/null 2>&1 || exit 1
        fi
        mkdir -p "$(dirname "$kgs_root")"
        exec 9>"$(dirname "$kgs_root")/.deployment.lock"
        flock 9
        if [ ! -e "$kgs_root" ]; then
            staging=$(mktemp -d "$(dirname "$kgs_root")/kg-checkout-XXXXXX")
            git clone --no-checkout "$2" "$staging/repository" >/dev/null 2>&1
            git -C "$staging/repository" checkout --detach "$4" >/dev/null 2>&1
            mv "$staging/repository" "$kgs_root"
            rmdir "$staging"
        fi
    )
fi
[ "$(git -C "$kgs_root" rev-parse HEAD)" = "$4" ] || { echo "KGS checkout HEAD mismatch" >&2; exit 1; }
[ "$(git -C "$kgs_root" remote get-url origin)" = "$2" ] || { echo "KGS checkout origin mismatch" >&2; exit 1; }
[ -z "$(git -C "$kgs_root" status --porcelain)" ] || { echo "KGS checkout is not clean" >&2; exit 1; }
exec "$6" -u "$kgs_root/kernelgen_server/management.py" "$7" "$8"
"""
    if action == "compatibility":
        # Read the exact, clean checkout verified above; no local KGS required.
        bootstrap = bootstrap.replace(
            'exec "$6" -u "$kgs_root/kernelgen_server/management.py" "$7" "$8"',
            'exec "$6" -c \'import json, pathlib, sys; '
            'print("' + RESULT_MARKER + '" + json.dumps({"text": '
            'pathlib.Path(sys.argv[1]).read_text()}))\' "$kgs_root/compatibility.yaml"',
        )
    return [
        "/bin/bash", "--noprofile", "--norc", "-c", bootstrap, "kg-kgs-bootstrap",
        payload["kgs_root"], payload["repository"], payload["release"], payload["commit"],
        payload.get("env_file", ""), remote_python, action, encoded,
    ]


def _remote_call(
    *,
    ssh_command: list[str],
    container: str,
    remote_python: str,
    action: str,
    payload: dict,
) -> dict:
    command = _remote_exec_command(
        ssh_command,
        container=container,
        command=_remote_management_command(remote_python, action, payload),
    )
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(
            f"remote KGS {action} failed with exit code {result.returncode}{suffix}"
        )
    marker = RESULT_MARKER
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(marker):
            value = json.loads(line[len(marker) :])
            if not isinstance(value, dict):
                break
            return value
    raise RuntimeError(f"remote KGS {action} returned no structured result")


def _remote_follow_logs(
    *,
    ssh_command: list[str],
    container: str,
    remote_python: str,
    payload: dict,
) -> int:
    command = _remote_exec_command(
        ssh_command,
        container=container,
        command=_remote_management_command(remote_python, "follow_logs", payload),
    )
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"remote KGS log stream exited with code {return_code}")
    return 0


def _git_value(root: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(root), *arguments])


def _validate_checkout(root: Path, *, repository: str, commit: str) -> None:
    if not (root / ".git").exists():
        raise ValueError(f"not a Git checkout: {root}")
    head = _git_value(root, "rev-parse", "HEAD")
    if head != commit:
        raise ValueError(f"checkout HEAD mismatch: expected {commit}, got {head}")
    if _git_value(root, "status", "--porcelain"):
        raise ValueError(f"checkout has uncommitted changes: {root}")
    origin = _git_value(root, "remote", "get-url", "origin")
    if origin != repository:
        raise ValueError(f"checkout origin does not match the locked repository: {root}")


def _prepare_checkout(
    root: Path,
    *,
    repository: str,
    release: str | None,
    branch: str | None,
    commit: str,
) -> None:
    if root.exists():
        _validate_checkout(root, repository=repository, commit=commit)
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="kg-checkout-", dir=str(root.parent)))
    checkout = temporary_root / "repository"
    try:
        command = ["git", "clone"]
        reference = release or branch
        if reference:
            command.extend(["--branch", reference, "--single-branch"])
        command.extend([repository, str(checkout)])
        _run(command)
        _run(["git", "checkout", "--detach", commit], cwd=checkout)
        _validate_checkout(checkout, repository=repository, commit=commit)
        os.replace(checkout, root)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _package_versions(python: str) -> dict:
    script = """
import importlib.metadata
import json
names = [
    'torch', 'triton', 'torch-npu', 'torch-mlu', 'torch-musa', 'torch-gcu',
    'kernelgen-server-client', 'fastapi', 'uvicorn', 'pydantic', 'requests', 'safetensors',
]
values = {}
for name in names:
    key = name.replace('-', '_')
    try:
        values[key] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        values[key] = None
print(json.dumps(values))
"""
    result = _run([python, "-c", script])
    return json.loads(result)


def _validate_imports(python: str, modules: list[str]) -> dict:
    script = """
import importlib
import json
import sys
values = {}
for name in json.loads(sys.argv[1]):
    module = importlib.import_module(name)
    values[name] = str(getattr(module, '__version__', 'installed'))
print(json.dumps(values))
"""
    return json.loads(_run([python, "-c", script, json.dumps(modules)]))


def _validate_local_server_install(python: str, kgs_root: Path) -> None:
    script = """
import importlib
from pathlib import Path
for name in ('kernelgen_client', 'kernelgen_server', 'fastapi', 'uvicorn', 'safetensors'):
    module = importlib.import_module(name)
    if name in ('kernelgen_client', 'kernelgen_server'):
        print(name + ':' + str(Path(module.__file__).resolve()))
"""
    paths = dict(line.split(":", 1) for line in _run([python, "-c", script]).splitlines())
    expected = {"kernelgen_server": kgs_root, "kernelgen_client": kgs_root / "client"}
    for name, root in expected.items():
        module_path = Path(paths[name])
        if root not in module_path.parents:
            raise RuntimeError(f"{name} imports from {module_path}, expected checkout {root}")


def _resolve_python(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.absolute()
    else:
        found = shutil.which(value)
        if found is None:
            raise FileNotFoundError(f"Python executable not found: {value}")
        resolved = Path(found).absolute()
    if not resolved.is_file():
        raise FileNotFoundError(f"Python executable not found: {resolved}")
    return str(resolved)


def _required_runtime_modules(backend: str) -> list[str]:
    modules = ["torch", "triton", "fastapi", "uvicorn", "safetensors"]
    backend_module = {
        "enflame": "torch_gcu",
        "npu": "torch_npu",
        "mlu": "torch_mlu",
        "musa": "torch_musa",
    }.get(backend)
    if backend_module:
        modules.append(backend_module)
    return modules


def _install_server(python: str, root: Path) -> dict:
    before = _package_versions(python)
    _run([python, "-m", "pip", "install", "-e", str(root / "client")])
    target = f"{root}[server]"
    command = [
        python,
        "-m",
        "pip",
        "install",
        "-e",
        target,
    ]
    _run(command)
    after = _package_versions(python)
    core = ("torch", "triton", "torch_npu", "torch_mlu", "torch_musa", "torch_gcu")
    changed = [name for name in core if before.get(name) != after.get(name)]
    if changed:
        raise RuntimeError(
            "KGS installation changed protected runtime packages: " + ", ".join(changed)
        )
    return {"before": before, "after": after}


def _kg_release() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as handle:
            version = tomllib.load(handle).get("project", {}).get("version")
        if version:
            return f"v{version}"
    return f"v{importlib.metadata.version('kernelgen')}"


def _compatibility(compatibility: dict) -> tuple[str, str, list[str]]:
    schema = compatibility.get("schema_version")
    release = compatibility.get("server_release", {})
    if schema == 1:
        return (
            str(release.get("version", "")),
            str(release.get("api_version", "")),
            [str(item) for item in release.get("supported_api_versions", [])],
        )
    if schema == 2:
        return (
            str(release.get("version", "")),
            str(release.get("protocol_version", "")),
            [str(item) for item in compatibility.get("supported_catalog_api_versions", [])],
        )
    raise ValueError(f"unsupported KGS compatibility schema: {schema}")


def _flaggems_spec(config: dict) -> dict[str, str | None]:
    compatibility_path = Path(config["kgs_root"]) / "compatibility.yaml"
    compatibility = (yaml.safe_load(_remote_call_config(config, "compatibility")["text"])
                     if config["target"] == "remote" else _read_yaml(compatibility_path))
    framework = compatibility.get("frameworks", {}).get("flaggems")
    if not isinstance(framework, dict):
        raise ValueError(
            f"KGS compatibility manifest does not declare frameworks.flaggems: "
            f"{compatibility_path}"
        )
    policy = framework.get("revision_policy")
    if policy not in {"exact", "branch"}:
        raise ValueError("KGS FlagGems revision_policy must be exact or branch")
    if policy == "branch" and framework.get("revision") is not None:
        raise ValueError("branch-based FlagGems selection must not also declare a fixed revision")
    values = {
        "repository": framework.get("repository"),
        "branch": framework.get("branch"),
        "commit": framework.get("revision"),
    }
    missing = [
        key
        for key, value in values.items()
        if (key != "commit" or policy == "exact") and (not isinstance(value, str) or not value)
    ]
    if missing:
        raise ValueError(
            "KGS FlagGems declaration is missing: " + ", ".join(sorted(missing))
        )
    return values


def _configured_flaggems(config: dict) -> dict | None:
    root = config.get("flaggems_root")
    if root is None:
        return None
    spec = _flaggems_spec(config)
    commit = config.get("flaggems_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("configured FlagGems commit must be a full lowercase commit hash")
    return {
        **spec,
        "commit": commit,
        "branch": config.get("flaggems_branch", spec["branch"]),
        "root": str(root),
    }


def _resolve_flaggems_revision(repository: str, revision: str) -> tuple[str, str | None]:
    """Resolve an explicitly selected branch once; full commits need no lookup."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision, None
    if not revision or revision.startswith("-") or any(c.isspace() or c == "\0" for c in revision):
        raise ValueError("FlagGems revision must be a branch name or full lowercase commit hash")
    _run(["git", "check-ref-format", f"refs/heads/{revision}"])
    reference = f"refs/heads/{revision}"
    rows = _run_git_remote(["git", "ls-remote", "--exit-code", repository, reference]).splitlines()
    matches = [row.split() for row in rows if len(row.split()) == 2 and row.split()[1] == reference]
    if len(matches) != 1 or not re.fullmatch(r"[0-9a-f]{40}", matches[0][0]):
        raise ValueError(f"cannot resolve FlagGems branch: {revision}")
    return matches[0][0], revision


def _load_config(instance: str) -> dict:
    raw = read_json(_config_path(instance))
    if raw is None:
        raise FileNotFoundError(
            f"server instance {instance!r} does not exist; provide its deployment options "
            f"with kg server start {instance}"
        )
    return raw


def _save_config(instance: str, config: dict) -> None:
    atomic_write_json(_config_path(instance), config)


def _load_process(instance: str) -> ServerProcessRecord | None:
    raw = read_json(_process_path(instance))
    return None if raw is None else ServerProcessRecord.model_validate(raw)


def _active_process(instance: str) -> ServerProcessRecord | None:
    process = _load_process(instance)
    if process is None:
        return None
    return process if process_is_alive(process.pid, process.process_start) else None


def _server_url(config: dict) -> str:
    port = config.get("listen_port", config["port"])
    return f"http://127.0.0.1:{port}"


def _http_status(
    config: dict,
    *,
    timeout: float = 2.0,
) -> dict:
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f"{_server_url(config)}/status", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"KGS status request failed: {exc}") from exc


def _remote_payload(
    config: dict,
    *,
    include_flaggems: bool = False,
) -> dict:
    kgs_lock = _kgs_lock()["kgs"]
    payload = {
        "repository": config.get("kgs_repository", kgs_lock["repository"]),
        "release": config.get("kgs_version", config["kgs_release"]),
        "commit": config["kgs_commit"],
        "backend": config["backend"],
        "devices": config["devices"],
        "timing": config["timing"],
        "port": config["port"],
        "max_workers": config["max_workers"],
        "kgs_root": config["remote_kgs_root"],
        "state_root": config["remote_state_root"],
    }
    if config.get("remote_env_file") is not None:
        payload["env_file"] = config["remote_env_file"]
    if include_flaggems:
        flaggems = _configured_flaggems(config)
        if flaggems is not None:
            payload["flaggems"] = flaggems
    return payload


def _remote_call_config(
    config: dict,
    action: str,
    additions: dict | None = None,
) -> dict:
    if config.get("target") != "remote":
        raise ValueError("managed KGS target is not remote")
    payload = _remote_payload(
        config,
        include_flaggems=action in {"start", "doctor"},
    )
    if additions:
        payload.update(additions)
    return _remote_call(
        ssh_command=list(config["ssh_command"]),
        container=str(config["container"]),
        remote_python=str(config["remote_python"]),
        action=action,
        payload=payload,
    )


def _live_status_problems(
    config: dict,
    status: dict,
    *,
    require_idle: bool,
) -> list[str]:
    problems: list[str] = []
    if status.get("api_version") != config["protocol_version"]:
        problems.append("running KGS Protocol does not match instance configuration")
    if config.get("strict_kgs_release", True) and (
        status.get("server_version") != config["kgs_release"]
    ):
        problems.append("running KGS release does not match instance configuration")
    if status.get("backend") != config["backend"]:
        problems.append("running KGS backend does not match instance configuration")
    if status.get("workers") != config["max_workers"]:
        problems.append("running KGS worker count does not match instance configuration")
    if config.get("flaggems_root") and "flaggems" not in (
        status.get("evaluation_adapters") or []
    ):
        problems.append("running KGS does not provide the installed FlagGems adapter")
    scheduler = status.get("scheduler", {})
    device_count = len(config["devices"])
    if len(status.get("devices", [])) != device_count:
        problems.append("running KGS visible device count does not match configuration")
    if not (
        scheduler.get("device_slots") == device_count
        and scheduler.get("healthy") == device_count
        and scheduler.get("checking") == 0
        and scheduler.get("broken") == 0
        and scheduler.get("max_active", device_count) <= device_count
    ):
        problems.append("KGS scheduler device slots are not healthy")
    if require_idle and not (
        scheduler.get("active") == 0
        and scheduler.get("waiting") == 0
        and scheduler.get("available") == device_count
    ):
        problems.append("KGS scheduler is not idle after startup")
    return problems


def _parse_devices(value: str | list[str]) -> list[str]:
    items = value if isinstance(value, list) else value.split(",")
    devices = [str(item).strip() for item in items if str(item).strip()]
    if not devices:
        raise ValueError("--devices must name at least one visible device")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate device tokens")
    return devices


def _validate_port(value: int, option: str) -> int:
    if not 1 <= value <= 65535:
        raise ValueError(f"{option} must be between 1 and 65535")
    return value


def _validate_remote_python(value: str) -> str:
    if not value or any(character in value for character in ("\n", "\r", "\0")):
        raise ValueError("--remote-python must name one executable")
    return value


def _validate_remote_env_file(value: str) -> str:
    if not value or any(character in value for character in ("\n", "\r", "\0")):
        raise ValueError("--remote-env-file must name one file")
    return value


def _create_instance_config(args, instance: str) -> dict:
    target = getattr(args, "target", None)
    backend = getattr(args, "backend", None)
    devices_value = getattr(args, "devices", None)
    kgs_version_value = getattr(args, "kgs_version", None)
    remote_env_file_value = getattr(args, "remote_env_file", None)
    missing = [
        option
        for option, value in (
            ("--target", target),
            ("--backend", backend),
            ("--devices", devices_value),
        )
        if value is None
    ]
    if missing:
        raise ValueError("first start requires " + ", ".join(missing))
    assert target in {"local", "remote"}
    devices = _parse_devices(devices_value)
    max_workers = len(devices) if args.max_workers is None else args.max_workers
    if max_workers < 1:
        raise ValueError("--max-workers must be positive")
    timing = args.timing or "auto"
    if target == "local":
        remote_only = {
            "--remote-port": args.remote_port,
            "--listen-port": args.listen_port,
            "--max-streams": args.max_streams,
            "--ssh-command": args.ssh_command,
            "--container": args.container,
            "--remote-python": args.remote_python,
            "--remote-kgs-root": args.remote_kgs_root,
            "--remote-state-root": args.remote_state_root,
            "--remote-env-file": remote_env_file_value,
        }
        supplied = [option for option, value in remote_only.items() if value is not None]
        if supplied:
            raise ValueError(f"{', '.join(supplied)} can only be used with --target remote")
        port = _validate_port(8000 if args.port is None else args.port, "--port")
        listen_port = port
        ssh_command: list[str] = []
        container = None
        remote_python = None
        remote_kgs_root = None
        remote_state_root = None
        remote_env_file = None
        max_streams = None
    else:
        if args.port is not None:
            raise ValueError("--port is for local instances; use --remote-port")
        required = [
            option
            for option, value in (
                ("--ssh-command", args.ssh_command),
                ("--container", args.container),
                ("--remote-port", args.remote_port),
                ("--listen-port", args.listen_port),
            )
            if value is None
        ]
        if required:
            raise ValueError("first remote start requires " + ", ".join(required))
        port = _validate_port(args.remote_port, "--remote-port")
        listen_port = _validate_port(args.listen_port, "--listen-port")
        ssh_command = _parse_ssh_command(args.ssh_command)
        container = _validate_container(args.container)
        remote_python = _validate_remote_python(args.remote_python or "python3")
        remote_kgs_root = args.remote_kgs_root
        remote_state_root = args.remote_state_root or f"~/.kernelgen/servers/{instance}"
        remote_env_file = (
            _validate_remote_env_file(remote_env_file_value)
            if remote_env_file_value is not None
            else None
        )
        max_streams = max_workers + 2 if args.max_streams is None else args.max_streams
        if max_streams < 1:
            raise ValueError("--max-streams must be positive")

    lock = _kgs_lock()
    if lock.get("kg_release") != _kg_release():
        raise ValueError(
            f"deployment lock is for {lock.get('kg_release')}, installed KG is {_kg_release()}"
        )
    python = _resolve_python(args.python or sys.executable)
    selected_kgs = getattr(args, "resolved_kgs", None) or _resolve_kgs_version(
        kgs_version_value
    )
    kgs_root = (
        args.kgs_root.expanduser().resolve()
        if args.kgs_root is not None
        else cli_home() / "kgs" / str(selected_kgs["commit"])
    )
    if target == "remote":
        # Remote deployment owns the KGS checkout; local KG records the
        # user-facing version and its internally pinned source identity.
        kgs_release = str(selected_kgs["release"])
        protocol = str(lock["validated_protocol"])
        remote_kgs_root = remote_kgs_root or (
            f"~/.kernelgen/kgs/{selected_kgs['commit']}"
        )
        strict_kgs_release = bool(selected_kgs["strict_release"])
    else:
        _prepare_checkout(
            kgs_root,
            repository=str(selected_kgs["repository"]),
            release=None,
            branch=(
                str(selected_kgs["branch"])
                if selected_kgs["branch"] is not None
                else None
            ),
            commit=str(selected_kgs["commit"]),
        )
        compatibility = _read_yaml(kgs_root / "compatibility.yaml")
        kgs_release, protocol, _ = _compatibility(compatibility)
        strict_kgs_release = True
    if protocol != lock["validated_protocol"]:
        raise ValueError("selected KGS protocol does not match the KG deployment lock")
    if bool(selected_kgs["strict_release"]) and kgs_release != selected_kgs["release"]:
        raise ValueError("KGS compatibility manifest does not match the KG deployment lock")

    config = {
        "schema_version": "1.0",
        "instance": instance,
        "target": target,
        "kg_release": _kg_release(),
        "kgs_version": str(selected_kgs["version"]),
        "kgs_release": kgs_release,
        "kgs_commit": str(selected_kgs["commit"]),
        "kgs_branch": selected_kgs["branch"],
        "kgs_repository": str(selected_kgs["repository"]),
        "strict_kgs_release": strict_kgs_release,
        "protocol_version": protocol,
        "kgs_root": str(kgs_root),
        "backend": backend,
        "devices": devices,
        "timing": timing,
        "port": port,
        "listen_port": listen_port,
        "max_workers": max_workers,
        "python": python,
        "ssh_command": ssh_command,
        "container": container,
        "remote_python": remote_python,
        "remote_kgs_root": remote_kgs_root,
        "remote_state_root": remote_state_root,
        "remote_env_file": remote_env_file,
        "max_streams": max_streams,
        "package_versions": {},
        "package_changes": {},
        "log_path": str(_log_path(instance)),
    }
    _save_config(instance, config)
    print(f"kgs_root: {kgs_root}")
    print(f"kgs_version: {selected_kgs['version']}")
    print(f"protocol: {protocol}")
    print(f"config: {_config_path(instance)}")
    return config


def _validate_instance_checkout(config: dict) -> None:
    lock = _kgs_lock()
    if config.get("kg_release") != _kg_release():
        raise ValueError("server instance belongs to a different KG release")
    repository = config.get("kgs_repository", lock["kgs"]["repository"])
    commit = config.get("kgs_commit", lock["kgs"]["commit"])
    if not isinstance(repository, str) or not repository:
        raise ValueError("server instance has no KGS repository")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("server instance has no pinned KGS commit")
    if config.get("target") == "remote":
        return
    _validate_checkout(
        Path(config["kgs_root"]),
        repository=repository,
        commit=commit,
    )


def _assert_existing_config_matches(args, config: dict) -> None:
    checks: list[tuple[str, object, object]] = []
    if args.target is not None:
        checks.append(("--target", args.target, config["target"]))
    if args.backend is not None:
        checks.append(("--backend", args.backend, config["backend"]))
    if args.devices is not None:
        checks.append(("--devices", _parse_devices(args.devices), config["devices"]))
    if args.timing is not None:
        checks.append(("--timing", args.timing, config["timing"]))
    if args.max_workers is not None:
        checks.append(("--max-workers", args.max_workers, config["max_workers"]))
    if args.python is not None:
        checks.append(("--python", _resolve_python(args.python), config["python"]))
    if args.kgs_root is not None:
        checks.append(
            (
                "--kgs-root",
                str(args.kgs_root.expanduser().resolve()),
                config["kgs_root"],
            )
        )
    if getattr(args, "kgs_version", None) is not None:
        checks.append(
            (
                "--kgs-version",
                args.kgs_version,
                config.get("kgs_version", config.get("kgs_release")),
            )
        )
    if args.port is not None:
        stored = config["port"] if config["target"] == "local" else None
        checks.append(("--port", args.port, stored))
    if args.remote_port is not None:
        stored = config["port"] if config["target"] == "remote" else None
        checks.append(("--remote-port", args.remote_port, stored))
    if args.listen_port is not None:
        checks.append(("--listen-port", args.listen_port, config.get("listen_port")))
    if args.max_streams is not None:
        checks.append(("--max-streams", args.max_streams, config.get("max_streams")))
    if args.ssh_command is not None:
        checks.append(
            (
                "--ssh-command",
                _parse_ssh_command(args.ssh_command),
                config.get("ssh_command"),
            )
        )
    if args.container is not None:
        checks.append(
            ("--container", _validate_container(args.container), config.get("container"))
        )
    if args.remote_python is not None:
        checks.append(
            (
                "--remote-python",
                _validate_remote_python(args.remote_python),
                config.get("remote_python"),
            )
        )
    for option, attribute in (
        ("--remote-kgs-root", "remote_kgs_root"),
        ("--remote-state-root", "remote_state_root"),
        ("--remote-env-file", "remote_env_file"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            requested = (
                _validate_remote_env_file(value)
                if attribute == "remote_env_file"
                else value
            )
            checks.append((option, requested, config.get(attribute)))
    for option, requested, stored in checks:
        if requested != stored:
            raise ValueError(
                f"{option} differs from server instance {config['instance']!r}; "
                f"run kg server configure {config['instance']} while it is stopped"
            )


def _wait_until_ready(
    config: dict,
    record: ServerProcessRecord,
    process: subprocess.Popen,
    *,
    timeout: float,
) -> dict:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"managed {record.target} process exited with code {process.returncode}; "
                f"inspect {record.log_path}"
            )
        try:
            status = _http_status(config)
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(0.5)
            continue
        problems = _live_status_problems(config, status, require_idle=True)
        if problems:
            raise RuntimeError("; ".join(problems))
        if process.poll() is not None:
            raise RuntimeError(f"managed process exited during readiness; inspect {record.log_path}")
        return status
    raise RuntimeError(f"KGS did not become ready before timeout: {last_error}")


def _start_local(instance: str, args, config: dict) -> int:
    env = dict(os.environ)
    for names in DEVICE_ENVIRONMENTS.values():
        for name in names:
            env.pop(name, None)
    for name in DEVICE_ENVIRONMENTS.get(config["backend"], ()):
        env[name] = ",".join(config["devices"])
    flaggems = _configured_flaggems(config)
    if flaggems is not None:
        env["KGS_FLAGGEMS_ROOT"] = flaggems["root"]
    command = [
        config["python"],
        "-u",
        "-m",
        "kernelgen_server.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(config["port"]),
        "--backend",
        config["backend"],
        "--timing",
        config["timing"],
        "--max-workers",
        str(config["max_workers"]),
        "--profile-artifact-root",
        str(_server_root(instance) / "profiles"),
    ]
    _server_root(instance).mkdir(parents=True, exist_ok=True)
    with Path(os.devnull).open("rb") as stdin_handle, _log_path(instance).open(
        "ab", buffering=0
    ) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=config["kgs_root"],
            env=env,
            stdin=stdin_handle,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    identity = process_start_identity(process.pid)
    if identity is None:
        raise RuntimeError("KGS process exited before registration")
    record = ServerProcessRecord(
        instance=instance,
        target="local",
        pid=process.pid,
        process_start=identity,
        log_path=_log_path(instance),
        server_url=_server_url(config),
        kgs_release=config["kgs_release"],
        kgs_commit=config["kgs_commit"],
        protocol_version=config["protocol_version"],
        backend=config["backend"],
        devices=config["devices"],
        port=config["port"],
        max_workers=config["max_workers"],
    )
    atomic_write_json(_process_path(instance), record.model_dump(mode="json"))
    try:
        _wait_until_ready(
            config,
            record,
            process,
            timeout=float(args.startup_timeout),
        )
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise
    print(f"instance: {instance}")
    print("target: local")
    print(f"pid: {process.pid}")
    print(f"url: {record.server_url}")
    print("status: RUNNING")
    return 0


def _start_remote_server(instance: str, config: dict) -> dict:
    return _remote_call_config(config, "start")


def _wait_for_proxy_ready(descriptor: int, process: subprocess.Popen, timeout: float) -> None:
    """The child confirms its own socket bind and SSH setup, not another listener."""
    readable, _, _ = select.select([descriptor], [], [], timeout)
    if not readable:
        raise RuntimeError("SSH proxy startup timed out; inspect remote-proxy.log")
    if os.read(descriptor, 6) != b"READY\n" or process.poll() is not None:
        raise RuntimeError("SSH proxy failed before readiness; inspect remote-proxy.log")


def _start_remote_proxy(instance: str, args, config: dict, remote_result: dict) -> int:
    previous = _load_process(instance)
    if previous is not None:
        # Remote start has succeeded even if the replacement local proxy fails.
        # Keep stop's identity guard attached to this managed remote process.
        previous = previous.model_copy(update={
            "remote_pid": int(remote_result["pid"]),
            "remote_process_start": str(remote_result["process_start"]),
            "remote_log_path": str(remote_result["log_path"]),
        })
        atomic_write_json(_process_path(instance), previous.model_dump(mode="json"))
    ssh_command = list(config["ssh_command"])
    container = str(config["container"])
    remote_python = str(config["remote_python"])
    listen_port = int(config["listen_port"])
    proxy_script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "remote_server"
        / "ssh_stdio_http_mux_proxy.py"
    )
    proxy_command = [
        sys.executable,
        "-u",
        str(proxy_script),
        "--listen-port",
        str(listen_port),
        "--ssh-command",
        shlex.join(ssh_command),
        "--remote-port",
        str(config["port"]),
        "--remote-python",
        remote_python,
        "--container",
        container,
        "--max-streams",
        str(config["max_streams"]),
        "--server-alive-interval",
        "6",
        "--server-alive-count-max",
        "30",
    ]
    _server_root(instance).mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen | None = None
    try:
        deadline = time.monotonic() + float(args.startup_timeout)
        with ExitStack() as stack:
            read_fd, write_fd = os.pipe()
            ready_reader = stack.enter_context(os.fdopen(read_fd, "rb", buffering=0))
            ready_writer = stack.enter_context(os.fdopen(write_fd, "wb", buffering=0))
            with Path(os.devnull).open("rb") as stdin_handle, _proxy_log_path(instance).open(
                "ab", buffering=0
            ) as log_handle:
                process = subprocess.Popen(
                    [*proxy_command, "--ready-fd", str(write_fd)],
                    stdin=stdin_handle,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    pass_fds=(write_fd,),
                )
            ready_writer.close()
            _wait_for_proxy_ready(
                ready_reader.fileno(), process, max(0, deadline - time.monotonic())
            )
        identity = process_start_identity(process.pid)
        if identity is None:
            raise RuntimeError("remote KGS proxy exited before registration")
        record = ServerProcessRecord(
            instance=instance,
            target="remote",
            pid=process.pid,
            process_start=identity,
            log_path=_proxy_log_path(instance),
            server_url=f"http://127.0.0.1:{listen_port}",
            kgs_release=config["kgs_release"],
            kgs_commit=config["kgs_commit"],
            protocol_version=config["protocol_version"],
            backend=config["backend"],
            devices=config["devices"],
            port=config["port"],
            max_workers=config["max_workers"],
            remote_pid=int(remote_result["pid"]),
            remote_process_start=str(remote_result["process_start"]),
            remote_log_path=str(remote_result["log_path"]),
        )
        atomic_write_json(_process_path(instance), record.model_dump(mode="json"))
        _wait_until_ready(
            config,
            record,
            process,
            timeout=max(0, deadline - time.monotonic()),
        )
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        print(
            f"warning: remote KGS for {instance!r} remains running; "
            "rerun start to restore its proxy",
            file=sys.stderr,
        )
        raise
    assert process is not None
    print(f"instance: {instance}")
    print("target: remote")
    print(f"proxy_pid: {process.pid}")
    print(f"remote_pid: {remote_result['pid']}")
    print(f"url: {record.server_url}")
    print("status: RUNNING")
    return 0


def _terminate_managed_process(process: ServerProcessRecord, timeout: float = 10.0) -> bool:
    if not process_is_alive(process.pid, process.process_start):
        return False
    os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_alive(process.pid, process.process_start):
            return False
        time.sleep(0.2)
    os.killpg(process.pid, signal.SIGKILL)
    return True


def _print_already_running(instance: str, config: dict, process: ServerProcessRecord) -> int:
    print(f"instance: {instance}")
    print(f"target: {config['target']}")
    print(f"pid: {process.pid}")
    if process.remote_pid:
        print(f"remote_pid: {process.remote_pid}")
    print(f"url: {_server_url(config)}")
    print("status: ALREADY_RUNNING")
    return 0


def _endpoint_conflict(instance: str, config: dict) -> str | None:
    requested = _server_url(config)
    if not _servers_root().is_dir():
        return None
    for path in _servers_root().iterdir():
        if not path.is_dir() or path.name == instance:
            continue
        try:
            other_config = _load_config(path.name)
            other = _load_process(path.name)
        except (ValueError, OSError):
            continue
        other_running_locally = bool(
            other
            and process_is_alive(other.pid, other.process_start)
        )
        if other_running_locally and _server_url(other_config) == requested:
            return path.name
        overlapping_devices = bool(
            set(config["devices"]) & set(other_config.get("devices", []))
        )
        same_local_devices = (
            config["target"] == "local"
            and other_config.get("target") == "local"
            and config["backend"] == other_config.get("backend")
            and overlapping_devices
        )
        if same_local_devices and other_running_locally:
            return path.name
        same_remote_host = (
            config["target"] == "remote"
            and other_config.get("target") == "remote"
            and other_config.get("ssh_command") == config.get("ssh_command")
            and other_config.get("container") == config.get("container")
        )
        same_remote_endpoint = (
            same_remote_host
            and other_config.get("port") == config.get("port")
        )
        same_remote_devices = same_remote_host and overlapping_devices
        if same_remote_endpoint or same_remote_devices:
            if other_running_locally:
                return path.name
            try:
                if _remote_call_config(other_config, "status").get("running"):
                    return path.name
            except RuntimeError as exc:
                raise RuntimeError(
                    "cannot verify whether remote endpoint is used by instance "
                    f"{path.name!r}: {exc}"
                ) from exc
    return None


def _endpoint_lock_paths(config: dict) -> list[Path]:
    identities = [f"listen:{config.get('listen_port', config['port'])}"]
    if config["target"] == "local":
        identities.extend(
            f"local-device:{config['backend']}:{device}"
            for device in config["devices"]
        )
    else:
        remote_identity = json.dumps(
            [config["ssh_command"], config["container"]],
            separators=(",", ":"),
        )
        identities.append(
            f"remote-endpoint:{remote_identity}:{config['port']}"
        )
        identities.extend(
            f"remote-device:{remote_identity}:{device}"
            for device in config["devices"]
        )
    paths = {
        _server_locks_root()
        / "endpoints"
        / f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.lock"
        for identity in identities
    }
    return sorted(paths)


def _notify_fleet_daemon() -> bool:
    try:
        from kernelgen.service.fleet_daemon import request_scan

        return request_scan()
    except (OSError, RuntimeError, ValueError):
        return False


def _start_configured_instance(instance: str, args, config: dict) -> int:
    conflict = _endpoint_conflict(instance, config)
    if conflict:
        raise RuntimeError(
            f"KGS endpoint or device is already used by running instance {conflict!r}"
        )
    active = _active_process(instance)
    if config["target"] == "local":
        if active is not None:
            status = _http_status(config)
            problems = _live_status_problems(config, status, require_idle=False)
            if problems:
                raise RuntimeError("; ".join(problems))
            return _print_already_running(instance, config, active)
        return _start_local(instance, args, config)

    remote_status = _remote_call_config(config, "status")
    if config.get("proxy_owner") == "fleet":
        if not remote_status.get("running"):
            _start_remote_server(instance, config)
        if not _notify_fleet_daemon():
            raise RuntimeError(
                "Fleet daemon is not running; remote KGS has no local proxy"
            )
        print(f"instance: {instance}")
        print("target: remote")
        print("proxy_owner: fleet")
        print("status: WAITING_FOR_FLEET")
        return 0
    if remote_status.get("running"):
        if active is not None:
            try:
                status = _http_status(config)
                problems = _live_status_problems(config, status, require_idle=False)
                if not problems:
                    return _print_already_running(instance, config, active)
            except RuntimeError:
                pass
            _terminate_managed_process(active)
        return _start_remote_proxy(instance, args, config, remote_status)
    if active is not None:
        _terminate_managed_process(active)
    remote_result = _start_remote_server(instance, config)
    return _start_remote_proxy(instance, args, config, remote_result)


def _ensure_local_environment(config: dict) -> None:
    python = config["python"]
    kgs_root = Path(config["kgs_root"])
    before = _package_versions(python)
    try:
        _validate_local_server_install(python, kgs_root)
    except RuntimeError:
        with file_lock(_server_locks_root() / "deployment.lock"):
            try:
                _validate_local_server_install(python, kgs_root)
            except RuntimeError:
                _install_server(python, kgs_root)
            _validate_local_server_install(python, kgs_root)
    _validate_imports(python, _required_runtime_modules(config["backend"]))
    after = _package_versions(python)
    config["package_versions"] = after
    config["package_changes"] = {
        "before": before, "after": after,
        "added": sorted(name for name in after if after[name] is not None and before.get(name) is None),
    }
    _save_config(config["instance"], config)
    flaggems = _configured_flaggems(config)
    if flaggems is not None:
        _validate_checkout(
            Path(flaggems["root"]),
            repository=flaggems["repository"],
            commit=flaggems["commit"],
        )


def _command_start(args) -> int:
    instance = _validate_instance_name(args.name)
    if args.startup_timeout <= 0:
        raise ValueError("--startup-timeout must be positive")
    with file_lock(_server_root(instance) / "lifecycle.lock"):
        existing = read_json(_config_path(instance))
        if existing is None:
            with file_lock(_server_locks_root() / "deployment.lock"):
                config = _create_instance_config(args, instance)
        else:
            config = existing
            _assert_existing_config_matches(args, config)
        _validate_instance_checkout(config)
        if config["target"] == "local":
            _ensure_local_environment(config)
        configured_gems = _configured_flaggems(config)
        gems_default = _flaggems_spec(config) if configured_gems is not None else None
        track_gems = (gems_default is not None and gems_default["commit"] is None
                      and configured_gems["branch"] == gems_default["branch"])
        if track_gems or (getattr(args, "install_gems", False) and configured_gems is None):
            _install_flaggems_locked(instance, config, allow_running_noop=True)
        with ExitStack() as endpoint_locks:
            for path in _endpoint_lock_paths(config):
                endpoint_locks.enter_context(file_lock(path))
            return _start_configured_instance(instance, args, config)


def _public_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "ssh_command"}


def _server_status(instance: str) -> dict:
    config = _load_config(instance)
    process = _load_process(instance)
    process_alive = (
        process is not None
        and process_is_alive(process.pid, process.process_start)
    )
    result = {
        "schema_version": "1.0",
        "state": "RUNNING" if process_alive else "STOPPED",
        "url": _server_url(config),
        "instance": instance,
        "config": _public_config(config),
        "process": process.model_dump(mode="json") if process else None,
        "server": None,
        "remote": None,
    }
    if config["target"] == "remote":
        try:
            result["remote"] = _remote_call_config(config, "status")
        except (RuntimeError, ValueError) as exc:
            result["remote_error"] = str(exc)
            result["state"] = "UNREACHABLE"
            result["error"] = str(exc)
        if not process_alive and result["remote"] and result["remote"].get("running"):
            result["state"] = "UNREACHABLE"
            result["error"] = "remote KGS is running but its local SSH proxy is stopped"
        elif process_alive and result["remote"] and not result["remote"].get("running"):
            result["state"] = "UNREACHABLE"
            result["error"] = "local SSH proxy is running but remote KGS is stopped"
    if process_alive:
        try:
            result["server"] = _http_status(config)
        except RuntimeError as exc:
            result["state"] = "UNREACHABLE"
            result["error"] = str(exc)
    return result


def _command_status(args) -> int:
    status = _server_status(_validate_instance_name(args.name))
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"instance: {status['instance']}")
        print(f"state: {status['state']}")
        print(f"url: {status['url']}")
        if status["process"]:
            print(f"target: {status['process']['target']}")
            print(f"pid: {status['process']['pid']}")
            if status["process"].get("remote_pid"):
                print(f"remote_pid: {status['process']['remote_pid']}")
        if status["server"]:
            print(f"server_version: {status['server'].get('server_version')}")
            print(f"api_version: {status['server'].get('api_version')}")
            scheduler = status["server"].get("scheduler", {})
            print(
                "scheduler: "
                f"active={scheduler.get('active')} waiting={scheduler.get('waiting')} "
                f"healthy={scheduler.get('healthy')} broken={scheduler.get('broken')}"
            )
        elif status.get("error"):
            print(f"error: {status['error']}")
    return 0


def _command_logs(args) -> int:
    instance = _validate_instance_name(args.name)
    config = _load_config(instance)
    process = _load_process(instance)
    if config["target"] == "remote":
        if args.follow:
            payload = _remote_payload(config)
            payload["lines"] = args.lines
            return _remote_follow_logs(
                ssh_command=list(config["ssh_command"]),
                container=str(config["container"]),
                remote_python=str(config["remote_python"]),
                payload=payload,
            )
        result = _remote_call_config(config, "logs", {"lines": args.lines})
        data = base64.b64decode(result["data"])
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        return 0
    path = _log_path(instance)
    if not path.exists():
        raise FileNotFoundError(f"KGS log not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-args.lines:]:
        print(line)
    if not args.follow:
        return 0
    position = path.stat().st_size
    while _active_process(instance) is not None:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(position)
            chunk = handle.read()
            position = handle.tell()
        if chunk:
            print(chunk, end="", flush=True)
        time.sleep(0.5)
    return 0


def _command_stop(args) -> int:
    instance = _validate_instance_name(args.name)
    with file_lock(_server_root(instance) / "lifecycle.lock"):
        config = _load_config(instance)
        process = _load_process(instance)
        remote_forced = False
        if config["target"] == "remote":
            additions = {"timeout": args.timeout}
            if process is not None:
                additions.update(
                    expected_pid=process.remote_pid,
                    expected_process_start=process.remote_process_start,
                )
            result = _remote_call_config(config, "stop", additions)
            remote_forced = bool(result.get("forced"))
        if process is None or not process_is_alive(process.pid, process.process_start):
            print(f"instance: {instance}")
            print("status: STOPPED")
            return 0
        local_forced = _terminate_managed_process(process, args.timeout)
    suffixes = []
    if local_forced:
        suffixes.append("local forced")
    if remote_forced:
        suffixes.append("remote forced")
    suffix = f" ({', '.join(suffixes)})" if suffixes else ""
    print(f"instance: {instance}")
    print(f"status: STOPPED{suffix}")
    return 0


def _install_flaggems_locked(instance: str, config: dict, revision=None, *, allow_running_noop=False) -> None:
    """Prepare FlagGems while the caller holds the instance lifecycle lock."""
    active = _active_process(instance)
    if active is not None and not allow_running_noop:
        raise RuntimeError(
            f"stop server instance {instance!r} before installing FlagGems"
        )
    _validate_instance_checkout(config)
    remote_running = (config["target"] == "remote"
                      and _remote_call_config(config, "status").get("running"))
    if remote_running and not allow_running_noop:
        raise RuntimeError(f"stop remote server instance {instance!r} before installing FlagGems")
    configured = _configured_flaggems(config)
    spec = _flaggems_spec(config)
    if spec["commit"] is None and revision is None:
        revision = spec["branch"]
    elif configured is not None:
        spec = {key: configured[key] for key in ("repository", "branch", "commit")}
    if revision is not None:
        commit, branch = _resolve_flaggems_revision(spec["repository"], revision)
        spec = {**spec, "commit": commit, "branch": branch}
    if allow_running_noop and configured is not None and configured["commit"] == spec["commit"]:
        return
    if active is not None:
        raise RuntimeError(f"new FlagGems revision is available; stop server instance {instance!r} before updating it")
    if remote_running:
        raise RuntimeError(f"stop remote server instance {instance!r} before installing FlagGems")
    if config["target"] == "local":
        root = cli_home() / "frameworks" / "FlagGems" / spec["commit"]
        with file_lock(_server_locks_root() / "deployment.lock"):
            _prepare_checkout(
                root,
                repository=spec["repository"],
                release=None,
                branch=spec["branch"],
                commit=spec["commit"],
            )
    else:
        requested_root = f"~/.kernelgen/frameworks/FlagGems/{spec['commit']}"
        result = _remote_call_config(
            config,
            "install_flaggems",
            {"flaggems": {**spec, "root": requested_root}},
        )
        if result.get("commit") != spec["commit"] or not result.get("root"):
            raise RuntimeError("remote FlagGems installation returned invalid metadata")
        root = str(result["root"])

    config["flaggems_root"] = str(root)
    config["flaggems_commit"] = spec["commit"]
    config["flaggems_branch"] = spec["branch"]
    _save_config(instance, config)



def _command_install_flaggems(args) -> int:
    instance = _validate_instance_name(args.name)
    with file_lock(_server_root(instance) / "lifecycle.lock"):
        config = _load_config(instance)
        _install_flaggems_locked(instance, config, getattr(args, "revision", None))
    print(f"instance: {instance}")
    print(f"flaggems_root: {config['flaggems_root']}")
    print(f"flaggems_commit: {config['flaggems_commit']}")
    print("status: INSTALLED")
    return 0


def _command_doctor(args) -> int:
    instance = _validate_instance_name(args.name)
    config = _load_config(instance)
    problems: list[str] = []
    lock = _read_yaml(_lock_manifest_path())
    if config["target"] == "remote":
        try:
            _remote_call_config(config, "doctor")
        except (RuntimeError, ValueError) as exc:
            problems.append(str(exc))
    else:
        try:
            _validate_checkout(
                Path(config["kgs_root"]),
                repository=lock["kgs"]["repository"],
                commit=config["kgs_commit"],
            )
            _validate_local_server_install(
                config["python"],
                Path(config["kgs_root"]),
            )
            _validate_imports(
                config["python"],
                _required_runtime_modules(config["backend"]),
            )
            flaggems = _configured_flaggems(config)
            if flaggems is not None:
                _validate_checkout(
                    Path(flaggems["root"]),
                    repository=flaggems["repository"],
                    commit=flaggems["commit"],
                )
        except (RuntimeError, ValueError) as exc:
            problems.append(str(exc))
    process = _active_process(instance)
    if process is not None:
        try:
            status = _http_status(config)
        except RuntimeError as exc:
            problems.append(str(exc))
        else:
            problems.extend(
                _live_status_problems(config, status, require_idle=False)
            )
    if problems:
        print("status: FAILED")
        for problem in problems:
            print(f"problem: {problem}")
        return 1
    print("status: OK")
    print(f"instance: {instance}")
    print(f"kgs_version: {config.get('kgs_version', config['kgs_release'])}")
    print(f"protocol: {config['protocol_version']}")
    return 0


def _command_list(args) -> int:
    instances = []
    if _servers_root().is_dir():
        for path in sorted(_servers_root().iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not (path / "config.json").is_file():
                continue
            try:
                _validate_instance_name(path.name)
                config = _load_config(path.name)
                process = _active_process(path.name)
            except (OSError, ValueError):
                continue
            if process is not None:
                state = (
                    "PROXY_RUNNING"
                    if config["target"] == "remote"
                    else "RUNNING"
                )
            elif config["target"] == "remote":
                state = "REMOTE_UNKNOWN"
            else:
                state = "STOPPED"
            instances.append(
                {
                    "name": path.name,
                    "target": config["target"],
                    "backend": config["backend"],
                    "devices": config["devices"],
                    "state": state,
                    "url": _server_url(config),
                }
            )
    if args.json:
        print(
            json.dumps(
                {"schema_version": "1.0", "servers": instances},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for item in instances:
            print(
                f"{item['name']:<20} {item['state']:<16} {item['target']:<6} "
                f"{item['backend']:<10} {item['url']}"
            )
    return 0


def _command_configure(args) -> int:
    instance = _validate_instance_name(args.name)
    with file_lock(_server_root(instance) / "lifecycle.lock"):
        config = _load_config(instance)
        process = _active_process(instance)
        if process is not None:
            raise RuntimeError(f"stop server instance {instance!r} before configuring it")
        if config["target"] == "remote":
            remote_status = _remote_call_config(config, "status")
            if remote_status.get("running"):
                raise RuntimeError(
                    f"stop remote server instance {instance!r} before configuring it"
                )
        changed = False

        def update(key: str, value) -> None:
            nonlocal changed
            if value is not None and config.get(key) != value:
                config[key] = value
                changed = True

        update("backend", args.backend)
        if args.devices is not None:
            update("devices", _parse_devices(args.devices))
        update("timing", args.timing)
        if args.max_workers is not None:
            if args.max_workers < 1:
                raise ValueError("--max-workers must be positive")
            update("max_workers", args.max_workers)
        if config["target"] == "local":
            if any(
                value is not None
                for value in (
                    args.remote_port,
                    args.listen_port,
                    args.max_streams,
                    args.ssh_command,
                    args.container,
                    args.remote_python,
                    args.remote_kgs_root,
                    args.remote_state_root,
                    getattr(args, "remote_env_file", None),
                )
            ):
                raise ValueError("remote options cannot configure a local instance")
            if args.port is not None:
                update("port", _validate_port(args.port, "--port"))
                update("listen_port", config["port"])
        else:
            if args.port is not None:
                raise ValueError("--port is for local instances; use --remote-port")
            deployment_keys = {
                "ssh_command": (
                    _parse_ssh_command(args.ssh_command)
                    if args.ssh_command is not None
                    else None
                ),
                "container": (
                    _validate_container(args.container)
                    if args.container is not None
                    else None
                ),
                "remote_python": (
                    _validate_remote_python(args.remote_python)
                    if args.remote_python is not None
                    else None
                ),
                "remote_kgs_root": args.remote_kgs_root,
                "remote_state_root": args.remote_state_root,
                "remote_env_file": (
                    _validate_remote_env_file(args.remote_env_file)
                    if getattr(args, "remote_env_file", None) is not None
                    else None
                ),
            }
            for key, value in deployment_keys.items():
                update(key, value)
            if args.remote_port is not None:
                update("port", _validate_port(args.remote_port, "--remote-port"))
            if args.listen_port is not None:
                update("listen_port", _validate_port(args.listen_port, "--listen-port"))
            if args.max_streams is not None:
                if args.max_streams < 1:
                    raise ValueError("--max-streams must be positive")
                update("max_streams", args.max_streams)
        if not changed:
            print(f"instance: {instance}")
            print("status: UNCHANGED")
            return 0
        _save_config(instance, config)
    print(f"instance: {instance}")
    print(f"config: {_config_path(instance)}")
    print("status: CONFIGURED")
    return 0


def _add_common_configuration_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=sorted(DEVICE_ENVIRONMENTS))
    parser.add_argument("--devices")
    parser.add_argument("--timing", choices=["auto", "triton", "profiler", "walltime"])
    parser.add_argument("--port", type=int, help="local KGS loopback port")
    parser.add_argument("--remote-port", type=int, help="remote KGS loopback port")
    parser.add_argument("--listen-port", type=int, help="local loopback proxy port")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-streams", type=int, help="SSH proxy logical stream limit")
    parser.add_argument("--ssh-command", help="base SSH connection command")
    parser.add_argument("--container", help="remote Docker container name, or '-' for the host")
    parser.add_argument("--remote-python")
    parser.add_argument("--remote-kgs-root")
    parser.add_argument("--remote-state-root")
    parser.add_argument(
        "--remote-env-file",
        help="remote env file used only while preparing KGS and FlagGems",
    )


def add_server_parser(subparsers) -> None:
    server = subparsers.add_parser("server", help="manage named local or remote KernelGen Servers")
    commands = server.add_subparsers(dest="server_command", required=True)

    start = commands.add_parser("start", help="deploy if needed and start a named KGS")
    start.add_argument("name")
    start.add_argument("--target", choices=["local", "remote"])
    _add_common_configuration_options(start)
    start.add_argument("--python")
    start.add_argument("--kgs-root", type=Path)
    start.add_argument(
        "--kgs-version",
        help="KGS release label or branch name; branch heads are resolved once",
    )
    start.add_argument("--startup-timeout", type=float, default=120.0)
    start.add_argument("--install-gems", "--install_gems", action="store_true",
                       help="prepare pinned FlagGems before starting; reuse an existing configured checkout")
    start.set_defaults(handler=_command_start)

    listing = commands.add_parser("list", help="list configured KGS instances without opening SSH")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=_command_list)

    configure = commands.add_parser("configure", help="change a stopped KGS instance")
    configure.add_argument("name")
    _add_common_configuration_options(configure)
    configure.set_defaults(handler=_command_configure)

    status = commands.add_parser("status", help="show KGS and scheduler status")
    status.add_argument("name")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_command_status)

    logs = commands.add_parser("logs", help="show KGS process output")
    logs.add_argument("name")
    logs.add_argument("--follow", "-f", action="store_true")
    logs.add_argument("--lines", type=int, default=200)
    logs.set_defaults(handler=_command_logs)

    stop = commands.add_parser("stop", help="stop a managed KGS instance")
    stop.add_argument("name")
    stop.add_argument("--timeout", type=float, default=10.0)
    stop.set_defaults(handler=_command_stop)

    install_flaggems = commands.add_parser(
        "install-flaggems",
        help="prepare a pinned FlagGems checkout for a stopped instance",
    )
    install_flaggems.add_argument("name")
    install_flaggems.add_argument(
        "--revision",
        help="FlagGems branch or full commit; resolve once and pin for this instance",
    )
    install_flaggems.set_defaults(handler=_command_install_flaggems)

    doctor = commands.add_parser("doctor", help="validate checkout, imports, and live status")
    doctor.add_argument("name")
    doctor.set_defaults(handler=_command_doctor)
