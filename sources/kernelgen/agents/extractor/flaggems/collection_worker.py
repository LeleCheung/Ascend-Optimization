"""Standalone stdlib collector supervisor, sent to the extraction container.

stdin: one JSON request followed by connection-liveness lines. stdout: evidence
records only. No KG, model CLI, KGS installation or shared mount is required.
"""

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tempfile
import threading


MAX_BYTES = 128 * 1024 * 1024
ARTIFACTS = ("pytest.log", "cases.json", "executed-source.json")


def snapshot_digest(files):
    hashes = {name: data if isinstance(data, dict) else hashlib.sha256(base64.b64decode(data, validate=True)).hexdigest()
              for name, data in files.items()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def excluded_source_path(path):
    """Never read/send environment files, Git metadata or execution caches."""
    return any(p in {".git", "__pycache__", ".pytest_cache", "env.sh", "env.opus.sh"}
               or p.startswith(".env") for p in path.parts)


def source_files(root):
    files = {}
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if excluded_source_path(relative):
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("source path escapes checkout")
        if excluded_source_path(path.resolve().relative_to(root.resolve())):
            raise ValueError("source link points into excluded metadata or environment file")
        if path.is_dir():
            if path.is_symlink():
                raise ValueError("source directory symlinks are unsupported")
            continue
        if not path.is_file():
            raise ValueError("source contains non-regular file")
        total += path.stat().st_size
        if total > MAX_BYTES:
            raise ValueError("source snapshot exceeds 128 MiB")
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise ValueError("absolute source symlinks are unsupported")
            files[relative.as_posix()] = {"symlink": target}
        else:
            files[relative.as_posix()] = base64.b64encode(path.read_bytes()).decode()
    return files


def write_files(root, files):
    total = 0
    links = []
    for name, encoded in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or excluded_source_path(path):
            raise ValueError("invalid snapshot path")
        target = root / path
        if isinstance(encoded, dict):
            link = encoded["symlink"]
            if Path(link).is_absolute() or not (target.parent / link).resolve().is_relative_to(root.resolve()):
                raise ValueError("invalid snapshot symlink")
            links.append((target, link))
            continue
        data = base64.b64decode(encoded, validate=True)
        total += len(data)
        if total > MAX_BYTES:
            raise ValueError("snapshot exceeds 128 MiB")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for target, link in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(link)


def clean_environment(root):
    allowed = {"PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "VIRTUAL_ENV", "LANG", "LC_ALL",
               "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONPATH=os.pathsep.join((str(root / "src"), str(root))))
    return env


def run(request, disconnected):
    root = Path(tempfile.mkdtemp(prefix="kg-case-collection-"))
    print(json.dumps({"event": "ready", "remote_workspace": str(root), "python": sys.executable}), flush=True)
    source, attempt = root / "source", root / "attempt"
    source.mkdir()
    attempt.mkdir()
    write_files(source, request["files"])
    expected = request["snapshot_sha256"]
    if snapshot_digest(source_files(source)) != expected:
        raise ValueError("transferred source snapshot mismatch")
    for name in ("collector.py", "test_collect_cases.py"):
        (attempt / name).write_text(request[name])
    (attempt / "request.json").write_text(json.dumps({**request["identity"], "source_root": str(source)}))
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--confcutdir", str(attempt), "test_collect_cases.py"]
    reason = ""
    with (attempt / "pytest.log").open("w") as log:
        process = subprocess.Popen(command, cwd=attempt, env=clean_environment(source),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finished = threading.Event()

        def supervise():
            nonlocal reason
            import time
            deadline = time.monotonic() + request["timeout"]
            while not finished.wait(0.2):
                if disconnected.is_set() or time.monotonic() >= deadline:
                    reason = "connection closed" if disconnected.is_set() else "collection timed out"
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return

        watcher = threading.Thread(target=supervise)
        watcher.start()
        try:
            code = process.wait()
        finally:
            finished.set()
            watcher.join()
            # Also reap descendants left behind by a completed/failed pytest.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    unchanged = snapshot_digest(source_files(source)) == expected
    artifacts = {}
    for name in ARTIFACTS:
        path = attempt / name
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
                raise ValueError("invalid or oversized collection artifact")
            artifacts[name] = base64.b64encode(path.read_bytes()).decode()
    result = {"event": "result", "returncode": code, "error": reason, "source_unchanged": unchanged,
              "snapshot_sha256": expected, "python": sys.executable, "remote_workspace": str(root)}
    # Immutable exit evidence remains available if the connection was cancelled.
    (root / "result.json").write_text(json.dumps(result))
    return {**result, "artifacts": artifacts}


def main():
    request = json.loads(sys.stdin.buffer.readline())
    disconnected = threading.Event()

    def monitor_input():
        while sys.stdin.buffer.readline():
            pass
        disconnected.set()

    threading.Thread(target=monitor_input, daemon=True).start()
    try:
        result = run(request, disconnected)
    except Exception as exc:
        result = {"event": "error", "error": str(exc)}
    print(json.dumps(result), flush=True)
    # Do not leave a daemon thread holding Python's buffered stdin at shutdown.
    os._exit(0 if result["event"] == "result" else 1)


if __name__ == "__main__":
    main()
