"""SSH stdio transport for the isolated Docker case-collection supervisor."""

import base64
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time

from kernelgen.data._atomic import atomic_write_json
from . import collection_worker
from .collection_config import ExtractionConfig


class CollectionTransportError(RuntimeError):
    """Infrastructure failures are not repaired by asking the model to rewrite pytest."""


def ssh_command(config: ExtractionConfig):
    command = ["docker", "exec", "-i", config.container, config.python, "-u", "-c",
               Path(collection_worker.__file__).read_text()]
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
            config.host, shlex.join(command)]


def snapshot(root):
    revision = None
    if (root / ".git").exists():
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "-z", "--untracked-files=all", "--ignored"], text=True)
        dirty = [entry for entry in status.split('\0') if entry and not
                 (entry.startswith('!! ') and collection_worker.excluded_source_path(Path(entry[3:])))]
        if dirty:
            raise ValueError("collection requires a clean source checkout")
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    files = collection_worker.source_files(root)
    return files, collection_worker.snapshot_digest(files), revision


def execute_pytest(attempt: Path, timeout: float, config: ExtractionConfig, *, checkpoint=lambda: None):
    """Keep stdin open until completion; EOF or remote timeout stops only this task."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("collection timeout must be positive")
    identity = json.loads((attempt / "request.json").read_text())
    root = Path(identity["source_root"])
    try:
        files, digest, revision = snapshot(root)
    except (OSError, ValueError) as exc:
        raise CollectionTransportError(str(exc)) from exc
    if "source_sha256" in identity and (identity["source_sha256"], identity.get("source_revision")) != (digest, revision):
        raise CollectionTransportError("source changed after collector generation")
    request = {"files": files, "snapshot_sha256": digest, "identity": identity, "timeout": timeout,
               **{name: (attempt / name).read_text() for name in ("collector.py", "test_collect_cases.py")}}
    payload = json.dumps(request).encode() + b"\n"
    record = {"execution": "ssh_docker_pytest", **config.model_dump(), "snapshot_sha256": digest,
              "source_revision": revision}
    atomic_write_json(attempt / "execution.json", record)
    checkpoint()
    stopped = threading.Event()
    output = attempt / "transport.jsonl"
    with output.open("wb") as stdout, (attempt / "transport.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(ssh_command(config), stdin=subprocess.PIPE, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
        except OSError as exc:
            raise CollectionTransportError("cannot start SSH collection; check SSH installation") from exc

        def send():
            try:
                process.stdin.write(payload)
                process.stdin.flush()
                while not stopped.wait(1):
                    process.stdin.write(b"alive\n")
                    process.stdin.flush()
            except (OSError, ValueError):
                pass

        writer = threading.Thread(target=send, daemon=True)
        writer.start()
        try:
            deadline = time.monotonic() + timeout + 45
            while process.poll() is None:
                checkpoint()
                if time.monotonic() >= deadline:
                    raise CollectionTransportError("SSH collection timed out; inspect execution.json and transport.log")
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stopped.set()
            if process.poll() is None:
                # Closing the connection signals the remote supervisor; its own
                # deadline remains effective even across a network partition.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            writer.join(timeout=5)
            try:
                process.stdin.close()
            except BrokenPipeError:
                # The peer may finish between the final heartbeat and close's
                # buffer flush. Judge success from the complete result below.
                pass
            # Keep the remote location even when cancellation interrupts recovery.
            try:
                with output.open() as events:
                    ready = json.loads(events.readline())
                if ready.get("event") == "ready":
                    record.update(remote_workspace=ready["remote_workspace"], python=ready["python"])
                    atomic_write_json(attempt / "execution.json", record)
            except (ValueError, KeyError):
                pass  # A failed connection may not have produced a ready record.
    try:
        messages = [json.loads(line) for line in output.read_text().splitlines()]
        if "remote_workspace" not in record:
            raise ValueError("missing remote ready record")
        result = messages[-1]
        if process.returncode or result.get("event") != "result":
            raise ValueError("remote supervisor failed")
        if result["snapshot_sha256"] != digest:
            raise ValueError("remote snapshot identity mismatch")
        artifacts = result["artifacts"]
        for name in collection_worker.ARTIFACTS:
            if name in artifacts:
                data = base64.b64decode(artifacts[name], validate=True)
                if len(data) > collection_worker.MAX_BYTES:
                    raise ValueError("oversized collection artifact")
                (attempt / name).write_bytes(data)
    except (ValueError, KeyError, IndexError) as exc:
        raise CollectionTransportError(f"incomplete remote collection; inspect {output} and transport.log") from exc
    if not result["source_unchanged"] or snapshot(root)[1:] != (digest, revision):
        raise CollectionTransportError("source changed during collection; preserve evidence and use a new workspace")
    if result["error"] or result["returncode"]:
        raise RuntimeError(f"temporary pytest failed ({result['error'] or result['returncode']}); inspect {attempt / 'pytest.log'}")
    if any(name not in artifacts for name in collection_worker.ARTIFACTS):
        raise RuntimeError(f"no timing cases collected (pytest may have skipped); inspect {attempt / 'pytest.log'}")
    return record
