"""Trusted, bounded debug-job execution on a KGS device slot."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

from kernelgen_client.debug.models import (
    DEFAULT_MAX_SOURCE_BYTES, DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_ARTIFACT_BYTES, DEFAULT_MAX_ARTIFACTS,
    _PROCESS_TERMINATION_GRACE_SECONDS, _INHERITED_ENV_NAMES,
    _INHERITED_ENV_PREFIXES, DebugArtifact, DebugExecutionResult,
    DebugJob, DebugJobRequest, DebugJobStatus, DebugSourceFile,
)
from kernelgen_client.protocol.version import KERNELGEN_API_VERSION
from ..runtime.backend import runtime_device_type
from ..runtime.environment import bind_backend_device
from ..runtime.process_control import terminate_process_tree


@dataclass
class _JobState:
    request: DebugJobRequest
    status: DebugJobStatus
    created_at: float
    device: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    output_truncated: bool = False
    artifacts: List[DebugArtifact] = field(default_factory=list)
    error: str = ""


class DebugJobStore:
    """Thread-safe in-memory job state with artifacts persisted on disk."""

    def __init__(
        self,
        root: Path,
        *,
        backend: str,
        max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
        retention_seconds: int = 24 * 60 * 60,
    ) -> None:
        self.root = Path(root).resolve()
        self.backend = backend
        self.max_source_bytes = max_source_bytes
        self.retention_seconds = retention_seconds
        self._jobs: Dict[str, _JobState] = {}
        self._lock = threading.Lock()

    def create(self, request: DebugJobRequest) -> DebugJob:
        if request.source_bytes() > self.max_source_bytes:
            raise ValueError(
                f"debug sources exceed {self.max_source_bytes} bytes"
            )
        self.prune()
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = _JobState(
                request=request,
                status="QUEUED",
                created_at=time.time(),
            )
            return self._snapshot_locked(job_id)

    def request(self, job_id: str) -> DebugJobRequest:
        with self._lock:
            return self._require_locked(job_id).request

    def begin(self, job_id: str, device: str) -> bool:
        with self._lock:
            state = self._require_locked(job_id)
            if state.status == "CANCELLED":
                return False
            if state.status != "QUEUED":
                raise RuntimeError(f"cannot start debug job in state {state.status}")
            state.status = "RUNNING"
            state.device = device
            state.started_at = time.time()
            return True

    def finish(self, job_id: str, result: DebugExecutionResult) -> DebugJob:
        with self._lock:
            state = self._require_locked(job_id)
            if state.status != "CANCELLED":
                state.status = result.status
            state.completed_at = time.time()
            state.exit_code = result.exit_code
            state.stdout = result.stdout
            state.stderr = result.stderr
            state.output_truncated = result.output_truncated
            state.artifacts = result.artifacts
            state.error = result.error
            return self._snapshot_locked(job_id)

    def fail(self, job_id: str, error: str) -> DebugJob:
        with self._lock:
            if self._require_locked(job_id).status == "CANCELLED":
                return self._snapshot_locked(job_id)
        return self.finish(
            job_id,
            DebugExecutionResult(
                status="FAILED",
                exit_code=None,
                stdout="",
                stderr="",
                output_truncated=False,
                artifacts=[],
                error=error,
            ),
        )

    def cancel(self, job_id: str) -> DebugJob:
        with self._lock:
            state = self._require_locked(job_id)
            if state.status in {"QUEUED", "RUNNING"}:
                state.status = "CANCELLED"
                state.completed_at = time.time()
            return self._snapshot_locked(job_id)

    def get(self, job_id: str) -> DebugJob:
        with self._lock:
            return self._snapshot_locked(job_id)

    def artifact_path(self, job_id: str, artifact_id: str) -> Path:
        with self._lock:
            state = self._require_locked(job_id)
            artifact = next(
                (item for item in state.artifacts if item.id == artifact_id),
                None,
            )
            if artifact is None:
                raise KeyError(artifact_id)
        artifact_root = (self.root / job_id / "artifacts").resolve()
        path = (artifact_root / artifact.path).resolve()
        if not path.is_file() or not path.is_relative_to(artifact_root):
            raise KeyError(artifact_id)
        return path

    def prune(self) -> None:
        cutoff = time.time() - self.retention_seconds
        expired: List[str] = []
        with self._lock:
            for job_id, state in self._jobs.items():
                if state.completed_at is not None and state.completed_at < cutoff:
                    expired.append(job_id)
            for job_id in expired:
                del self._jobs[job_id]
        for job_id in expired:
            shutil.rmtree(self.root / job_id, ignore_errors=True)

    def _require_locked(self, job_id: str) -> _JobState:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"debug job not found: {job_id}") from exc

    def _snapshot_locked(self, job_id: str) -> DebugJob:
        state = self._require_locked(job_id)
        return DebugJob(
            job_id=job_id,
            status=state.status,
            backend=self.backend,
            device=state.device,
            created_at=state.created_at,
            started_at=state.started_at,
            completed_at=state.completed_at,
            timeout_seconds=state.request.timeout_seconds,
            exit_code=state.exit_code,
            stdout=state.stdout,
            stderr=state.stderr,
            output_truncated=state.output_truncated,
            artifacts=state.artifacts,
            error=state.error,
        )


class LocalDebugJobRunner:
    """Execute trusted debug code in a fresh per-job working directory."""

    def __init__(
        self,
        root: Path,
        *,
        backend: str,
        catalog_root: Optional[Path] = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    ) -> None:
        self.root = Path(root).resolve()
        self.backend = backend
        self.catalog_root = Path(catalog_root).resolve() if catalog_root else None
        self.max_output_bytes = max_output_bytes
        self.max_artifact_bytes = max_artifact_bytes
        self.max_artifacts = max_artifacts
        self._processes: Dict[str, subprocess.Popen] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def run(
        self,
        job_id: str,
        request: DebugJobRequest,
        *,
        device: str,
    ) -> DebugExecutionResult:
        job_dir = (self.root / job_id).resolve()
        if not job_dir.is_relative_to(self.root):
            raise ValueError(f"unsafe debug job id: {job_id!r}")
        workspace = job_dir / "workspace"
        artifact_dir = job_dir / "artifacts"
        tmp_dir = job_dir / "tmp"
        home_dir = job_dir / "home"
        for path in (workspace, artifact_dir, tmp_dir, home_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._materialize_files(workspace, request.files)

        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        env = self._build_env(
            request,
            job_id=job_id,
            workspace=workspace,
            artifact_dir=artifact_dir,
            tmp_dir=tmp_dir,
            home_dir=home_dir,
            device=device,
        )
        start = time.monotonic()
        process: Optional[subprocess.Popen] = None
        exit_code: Optional[int] = None
        status: DebugJobStatus = "FAILED"
        error = ""
        with self._lock:
            cancel_event = self._cancel_events.setdefault(
                job_id, threading.Event()
            )
        try:
            if cancel_event.is_set():
                return self._cancelled_result()
            with (
                stdout_path.open("wb") as stdout_handle,
                stderr_path.open("wb") as stderr_handle,
            ):
                command = [
                    sys.executable if item == "{python}" else item
                    for item in request.command
                ]
                process = subprocess.Popen(
                    command,
                    cwd=workspace,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
                with self._lock:
                    self._processes[job_id] = process
                if cancel_event.is_set():
                    status = "CANCELLED"
                    error = "debug job cancelled"
                    self._terminate(process)

                while process.poll() is None:
                    if cancel_event.is_set():
                        status = "CANCELLED"
                        error = "debug job cancelled"
                        self._terminate(process)
                        break
                    if time.monotonic() - start >= request.timeout_seconds:
                        status = "TIMEOUT"
                        error = (
                            f"debug job exceeded {request.timeout_seconds} seconds"
                        )
                        self._terminate(process)
                        break
                    if (
                        self._combined_size(stdout_path, stderr_path)
                        > self.max_output_bytes
                    ):
                        status = "FAILED"
                        error = (
                            f"debug output exceeded {self.max_output_bytes} bytes"
                        )
                        self._terminate(process)
                        break
                    time.sleep(0.05)

                try:
                    exit_code = process.wait(
                        timeout=_PROCESS_TERMINATION_GRACE_SECONDS
                    )
                except subprocess.TimeoutExpired:
                    exit_code = process.returncode
                if cancel_event.is_set():
                    status = "CANCELLED"
                    error = "debug job cancelled"
                elif not error:
                    status = "SUCCEEDED" if exit_code == 0 else "FAILED"
                    if exit_code != 0:
                        error = f"command exited with status {exit_code}"
        except FileNotFoundError as exc:
            error = f"command not found: {exc.filename}"
        except Exception as exc:
            exit_code = process.returncode if process is not None else None
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                self._terminate(process)
            with self._lock:
                self._processes.pop(job_id, None)
                self._cancel_events.pop(job_id, None)

        stdout, stdout_truncated = self._read_bounded(stdout_path)
        stderr, stderr_truncated = self._read_bounded(stderr_path)
        artifacts, artifact_error = self._collect_artifacts(job_id, artifact_dir)
        if artifact_error:
            status = "FAILED"
            error = f"{error}; {artifact_error}".strip("; ")
        return DebugExecutionResult(
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            output_truncated=stdout_truncated or stderr_truncated,
            artifacts=artifacts,
            error=error,
        )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            cancel_event = self._cancel_events.setdefault(
                job_id, threading.Event()
            )
            cancel_event.set()
            process = self._processes.get(job_id)
        if process is None:
            return True
        self._terminate(process)
        return True

    def cancel_all(self) -> None:
        with self._lock:
            job_ids = set(self._cancel_events) | set(self._processes)
        for job_id in job_ids:
            self.cancel(job_id)

    @staticmethod
    def _cancelled_result() -> DebugExecutionResult:
        return DebugExecutionResult(
            status="CANCELLED",
            exit_code=None,
            stdout="",
            stderr="",
            output_truncated=False,
            artifacts=[],
            error="debug job cancelled",
        )

    def _materialize_files(
        self,
        workspace: Path,
        files: List[DebugSourceFile],
    ) -> None:
        root = workspace.resolve()
        for source in files:
            target = (root / source.path).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe debug source path: {source.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.content, encoding="utf-8")
            target.chmod(0o700 if source.executable else 0o600)

    def _build_env(
        self,
        request: DebugJobRequest,
        *,
        job_id: str,
        workspace: Path,
        artifact_dir: Path,
        tmp_dir: Path,
        home_dir: Path,
        device: str,
    ) -> Dict[str, str]:
        env = {
            name: value
            for name, value in os.environ.items()
            if name in _INHERITED_ENV_NAMES
            or name.startswith(_INHERITED_ENV_PREFIXES)
        }
        env.update(request.env)
        env.update(
            {
                "HOME": str(home_dir),
                "TMPDIR": str(tmp_dir),
                "KGS_BACKEND": self.backend,
                "KGS_DEBUG_JOB_ID": job_id,
                "KGS_DEBUG_WORKSPACE": str(workspace),
                "KGS_DEBUG_ARTIFACTS": str(artifact_dir),
                "KGS_ASSIGNED_DEVICE": device,
                "KGS_PYTHON": sys.executable,
                "KGS_DEVICE": f"{runtime_device_type(self.backend)}:0",
            }
        )
        if self.catalog_root is not None:
            env["KGS_CATALOG_ROOT"] = str(self.catalog_root)
        bind_backend_device(env, self.backend, device)
        return env

    @staticmethod
    def _combined_size(*paths: Path) -> int:
        return sum(path.stat().st_size for path in paths if path.exists())

    def _read_bounded(self, path: Path) -> tuple[str, bool]:
        if not path.is_file():
            return "", False
        data = path.read_bytes()
        truncated = len(data) > self.max_output_bytes
        if truncated:
            data = data[-self.max_output_bytes :]
        return data.decode("utf-8", errors="replace"), truncated

    def _collect_artifacts(
        self,
        job_id: str,
        artifact_dir: Path,
    ) -> tuple[List[DebugArtifact], str]:
        files = sorted(path for path in artifact_dir.rglob("*") if path.is_file())
        if len(files) > self.max_artifacts:
            return [], f"artifact count exceeds {self.max_artifacts}"
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes > self.max_artifact_bytes:
            return [], f"artifacts exceed {self.max_artifact_bytes} bytes"

        artifacts = []
        root = artifact_dir.resolve()
        manifest_files = {}
        for index, path in enumerate(files, start=1):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                return [], f"unsafe artifact path: {path}"
            relative = str(resolved.relative_to(root))
            artifact_id = f"a{index:03d}"
            media_type = (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            artifacts.append(
                DebugArtifact(
                    id=artifact_id,
                    path=relative,
                    media_type=media_type,
                    size_bytes=path.stat().st_size,
                    sha256=digest,
                    download_url=(
                        f"/debug/jobs/{job_id}/artifacts/{artifact_id}"
                    ),
                )
            )
            manifest_files[artifact_id] = relative

        manifest = {
            "api_version": KERNELGEN_API_VERSION,
            "job_id": job_id,
            "files": manifest_files,
        }
        (self.root / job_id / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return artifacts, ""

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        terminate_process_tree(
            process,
            grace_seconds=_PROCESS_TERMINATION_GRACE_SECONDS,
        )
