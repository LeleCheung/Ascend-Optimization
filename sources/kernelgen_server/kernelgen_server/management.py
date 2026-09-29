"""KGS-owned process management; executable before optional dependencies exist."""

from __future__ import annotations

import base64
from collections import deque
from contextlib import contextmanager
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fcntl

if __package__:
    from .process_utils import process_start_identity as _process_start, process_is_alive as _process_alive
else:
    # KG invokes this source file before pip has installed either distribution.
    # Load only this dependency-free primitive; importing kernelgen_client would
    # also import its Pydantic wire models before the client exists in the venv.
    from runpy import run_path

    _process_helpers = run_path(
        str(Path(__file__).resolve().parents[1] / "client/kernelgen_client/process_utils.py")
    )
    _process_start = _process_helpers["process_start_identity"]
    _process_alive = _process_helpers["process_is_alive"]


RESULT_MARKER = "__KERNELGEN_REMOTE_SERVER_RESULT_V1__"
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
PROTECTED_DISTRIBUTIONS = (
    "torch",
    "triton",
    "torch-npu",
    "torch-mlu",
    "torch-musa",
    "torch-gcu",
)


def _emit(value: dict) -> None:
    print(RESULT_MARKER + json.dumps(value, separators=(",", ":")), flush=True)


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"remote state must contain an object: {path}")
    return value


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        prefix = " ".join(command[:2])
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(
            f"remote command failed with exit code {result.returncode} ({prefix}){suffix}"
        )
    return result.stdout.strip()


@contextmanager
def _deployment_environment(payload: dict):
    """Temporarily load a machine-local shell environment for deployment only."""

    value = payload.get("env_file")
    if value is None:
        yield
        return
    if not isinstance(value, str) or not value or any(
        character in value for character in ("\n", "\r", "\0")
    ):
        raise ValueError("remote deployment env file must be one path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"remote deployment env file not found: {path}")
    result = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            'set -a; source "$1" >/dev/null || exit $?; env -0',
            "kg-deployment-env",
            str(path),
        ],
        cwd=str(path.parent),
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to load remote deployment env file: {path} "
            f"(exit code {result.returncode})"
        )
    loaded: dict[str, str] = {}
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        key, separator, item = entry.partition(b"=")
        if not separator:
            raise RuntimeError(f"remote deployment env file produced invalid output: {path}")
        loaded[os.fsdecode(key)] = os.fsdecode(item)
    original = dict(os.environ)
    os.environ.clear()
    os.environ.update(loaded)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _git_value(root: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(root), *arguments])


def _validate_checkout(root: Path, *, repository: str, commit: str) -> None:
    if not (root / ".git").is_dir():
        raise ValueError(f"not a Git checkout: {root}")
    head = _git_value(root, "rev-parse", "HEAD")
    if head != commit:
        raise ValueError(f"checkout HEAD mismatch: expected {commit}, got {head}")
    if _git_value(root, "status", "--porcelain"):
        raise ValueError(f"checkout has uncommitted changes: {root}")
    if _git_value(root, "remote", "get-url", "origin") != repository:
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


def _flaggems_checkout(payload: dict) -> Path | None:
    framework = payload.get("flaggems")
    if framework is None:
        return None
    if not isinstance(framework, dict):
        raise ValueError("remote FlagGems configuration must be an object")
    root = Path(framework["root"]).expanduser().resolve()
    _validate_checkout(
        root,
        repository=framework["repository"],
        commit=framework["commit"],
    )
    return root


def _install_flaggems(payload: dict) -> dict:
    framework = payload.get("flaggems")
    if not isinstance(framework, dict):
        raise ValueError("remote FlagGems installation requires a framework declaration")
    state_root, process_path, _ = _state_paths(payload)
    with _deployment_environment(payload):
        with _file_lock(state_root / "lifecycle.lock"):
            active = _active_record(process_path)
            if active is not None:
                raise RuntimeError(
                    f"remote KGS is already running with pid {active['pid']}"
                )
            root = Path(framework["root"]).expanduser().resolve()
            with _file_lock(root.parent / ".deployment.lock"):
                _prepare_checkout(
                    root,
                    repository=framework["repository"],
                    release=None,
                    branch=framework["branch"],
                    commit=framework["commit"],
                )
    return {"root": str(root), "commit": framework["commit"]}


def _package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in (*PROTECTED_DISTRIBUTIONS, "kernelgen-server-client", "fastapi", "uvicorn", "pydantic", "requests", "safetensors"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _runtime_modules(backend: str) -> list[str]:
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


def _validate_imports(modules: list[str]) -> dict[str, str]:
    script = """
import importlib
import json
import sys
versions = {}
for name in json.loads(sys.argv[1]):
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, '__version__', 'installed'))
print(json.dumps(versions))
"""
    return json.loads(
        _run([sys.executable, "-c", script, json.dumps(modules)])
    )


def _validate_server_import_root(kgs_root: Path) -> None:
    script = """
import importlib
from pathlib import Path
module = importlib.import_module('kernelgen_server')
print(Path(module.__file__).resolve())
"""
    module_path = Path(_run([sys.executable, "-c", script]))
    if kgs_root not in module_path.parents:
        raise RuntimeError(
            f"kernelgen_server imports from {module_path}, expected checkout {kgs_root}"
        )


def _validate_installed_server(kgs_root: Path) -> None:
    script = """
import importlib
from pathlib import Path
for name in ('kernelgen_client', 'kernelgen_server', 'fastapi', 'uvicorn', 'safetensors'):
    module = importlib.import_module(name)
    if name in ('kernelgen_client', 'kernelgen_server'):
        print(name + ':' + str(Path(module.__file__).resolve()))
"""
    paths = dict(line.split(":", 1) for line in _run([sys.executable, "-c", script]).splitlines())
    expected = {"kernelgen_server": kgs_root, "kernelgen_client": kgs_root / "client"}
    for name, root in expected.items():
        module_path = Path(paths[name])
        if root not in module_path.parents:
            raise RuntimeError(f"{name} imports from {module_path}, expected checkout {root}")


def _prepare(payload: dict) -> tuple[Path, dict[str, str]]:
    kgs_root = Path(payload["kgs_root"]).expanduser().resolve()
    with _deployment_environment(payload):
        with _file_lock(kgs_root.parent / ".deployment.lock"):
            _validate_checkout(kgs_root, repository=payload["repository"], commit=payload["commit"])
            try:
                _validate_installed_server(kgs_root)
                should_install = False
            except RuntimeError:
                should_install = True
            before = _package_versions()
            if should_install:
                _run([sys.executable, "-m", "pip", "install", "-e", str(kgs_root / "client")])
                target = f"{kgs_root}[server]"
                _run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-e",
                        target,
                    ]
                )
            after = _package_versions()
            changed = [
                name for name in PROTECTED_DISTRIBUTIONS if before[name] != after[name]
            ]
            if changed:
                raise RuntimeError(
                    "KGS installation changed protected runtime packages: "
                    + ", ".join(changed)
                )
            _atomic_write_json(Path(payload["state_root"]).expanduser() / "package-changes.json", {
                "before": before, "after": after,
                "added": sorted(name for name in after if after[name] is not None and before.get(name) is None),
            })
            versions = _validate_imports(
                ["kernelgen_client", "kernelgen_server", *_runtime_modules(payload["backend"])]
            )
            _validate_server_import_root(kgs_root)
    return kgs_root, versions


def _state_paths(payload: dict) -> tuple[Path, Path, Path]:
    root = Path(payload["state_root"]).expanduser().resolve()
    return root, root / "process.json", root / "kernelgen-server.log"


def _active_record(process_path: Path) -> dict | None:
    record = _read_json(process_path)
    if record is None:
        return None
    pid = int(record.get("pid", 0))
    process_start = str(record.get("process_start", ""))
    return record if pid > 0 and _process_alive(pid, process_start) else None


def _start(payload: dict) -> dict:
    state_root, process_path, log_path = _state_paths(payload)
    with _file_lock(state_root / "lifecycle.lock"):
        kgs_root, versions = _prepare(payload)
        active = _active_record(process_path)
        if active is not None:
            raise RuntimeError(f"remote KGS is already running with pid {active['pid']}")
        state_root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for names in DEVICE_ENVIRONMENTS.values():
            for name in names:
                env.pop(name, None)
        for name in DEVICE_ENVIRONMENTS.get(payload["backend"], ()):
            env[name] = ",".join(payload["devices"])
        flaggems_root = _flaggems_checkout(payload)
        if flaggems_root is not None:
            env["KGS_FLAGGEMS_ROOT"] = str(flaggems_root)
        command = [
            sys.executable,
            "-u",
            "-m",
            "kernelgen_server.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(payload["port"]),
            "--backend",
            payload["backend"],
            "--timing",
            payload["timing"],
            "--max-workers",
            str(payload["max_workers"]),
            "--profile-artifact-root",
            str(state_root / "profiles"),
        ]
        with Path(os.devnull).open("rb") as stdin_handle, log_path.open(
            "ab", buffering=0
        ) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(kgs_root),
                env=env,
                stdin=stdin_handle,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        identity = _process_start(process.pid)
        if identity is None:
            raise RuntimeError("remote KGS exited before registration")
        record = {
            "pid": process.pid,
            "process_start": identity,
            "log_path": str(log_path),
            "kgs_root": str(kgs_root),
            "started_at": time.time(),
        }
        _atomic_write_json(process_path, record)
    return {
        **record,
        "versions": versions,
    }


def _status(payload: dict) -> dict:
    _, process_path, log_path = _state_paths(payload)
    active = _active_record(process_path)
    return {
        "running": active is not None,
        "pid": active.get("pid") if active else None,
        "process_start": active.get("process_start") if active else None,
        "log_path": str(log_path),
    }


def _stop(payload: dict) -> dict:
    _, process_path, _ = _state_paths(payload)
    active = _active_record(process_path)
    if active is None:
        return {"stopped": True, "forced": False}
    expected_pid = payload.get("expected_pid")
    expected_start = payload.get("expected_process_start")
    if expected_pid is not None and int(active["pid"]) != int(expected_pid):
        raise RuntimeError("remote KGS pid no longer matches the managed process")
    if expected_start and active["process_start"] != expected_start:
        raise RuntimeError("remote KGS start identity no longer matches the managed process")
    pid = int(active["pid"])
    identity = str(active["process_start"])
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + float(payload.get("timeout", 10.0))
    while time.monotonic() < deadline:
        if not _process_alive(pid, identity):
            return {"stopped": True, "forced": False}
        time.sleep(0.2)
    os.killpg(pid, signal.SIGKILL)
    return {"stopped": True, "forced": True}


def _logs(payload: dict) -> dict:
    _, process_path, log_path = _state_paths(payload)
    offset = payload.get("offset")
    if not log_path.is_file():
        data = b""
        position = 0
    elif offset is None:
        lines = max(0, int(payload.get("lines", 200)))
        with log_path.open("rb") as handle:
            if lines:
                data = b"".join(deque(handle, maxlen=lines))
            else:
                handle.seek(0, os.SEEK_END)
                data = b""
            position = handle.tell()
    else:
        size = log_path.stat().st_size
        offset = max(0, min(int(offset), size))
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
            position = handle.tell()
    if len(data) > 2 * 1024 * 1024:
        data = data[-2 * 1024 * 1024 :]
    active = _active_record(process_path)
    return {
        "data": base64.b64encode(data).decode("ascii"),
        "position": position,
        "running": active is not None,
    }


def _follow_logs(payload: dict) -> None:
    _, process_path, log_path = _state_paths(payload)
    initial = _logs(payload)
    initial_data = base64.b64decode(initial["data"])
    if initial_data:
        sys.stdout.buffer.write(initial_data)
        sys.stdout.buffer.flush()
    position = int(initial["position"])
    while _active_record(process_path) is not None:
        time.sleep(0.5)
        size = log_path.stat().st_size if log_path.is_file() else 0
        if size < position:
            position = 0
        update = _logs({**payload, "offset": position})
        position = int(update["position"])
        data = base64.b64decode(update["data"])
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()


def _doctor(payload: dict) -> dict:
    kgs_root = Path(payload["kgs_root"]).expanduser().resolve()
    _validate_checkout(
        kgs_root,
        repository=payload["repository"],
        commit=payload["commit"],
    )
    versions = _validate_imports(["kernelgen_server", *_runtime_modules(payload["backend"])])
    _validate_server_import_root(kgs_root)
    _flaggems_checkout(payload)
    return {"ok": True, "versions": versions}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: management.py ACTION BASE64_JSON")
    action = sys.argv[1]
    payload = json.loads(base64.b64decode(sys.argv[2]).decode("utf-8"))
    if action == "follow_logs":
        _follow_logs(payload)
        return 0
    handlers = {
        "start": _start,
        "status": _status,
        "stop": _stop,
        "logs": _logs,
        "install_flaggems": _install_flaggems,
        "doctor": _doctor,
    }
    if action not in handlers:
        raise ValueError(f"unsupported remote action: {action}")
    _emit(handlers[action](payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
