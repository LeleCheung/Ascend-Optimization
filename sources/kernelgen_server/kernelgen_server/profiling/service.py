"""Profile execution and artifact-manifest management."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from ..runtime.operations import (
    OperationCancelled,
    raise_if_operation_cancelled,
)
from .models import (
    BackendProfileResult,
    ProfileArtifact,
    ProfileRequest,
    ProfileResult,
)
from .process import ProfileDeadlineExceeded, RequestDeadline
from .registry import get_profiler


class ProfileArtifactNotFound(KeyError):
    """A profile or artifact identifier does not resolve safely."""


class ProfileService:
    """Run backend profilers and own their persisted public artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        backend: str,
        hardware: Mapping[str, Any] | None = None,
        operator_bundle_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.hardware = dict(hardware or {})
        self.operator_bundle_root = operator_bundle_root

    def run(self, request: ProfileRequest, device: str) -> ProfileResult:
        started_at = time.monotonic()
        profile_id = uuid.uuid4().hex
        artifact_dir = self.root / profile_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        profiler = get_profiler(self.backend)
        effective_options = request.options
        option_warnings: list[str] = []
        try:
            raise_if_operation_cancelled()
            if not profiler.available_for(request.options):
                backend_result = BackendProfileResult(
                    status="unsupported",
                    profiler=profiler.name,
                    error=f"profiler {profiler.name!r} is unavailable",
                )
            elif request.options.level not in profiler.levels_for(request.options):
                backend_result = BackendProfileResult(
                    status="unsupported",
                    profiler=profiler.name,
                    error=(
                        f"profiling level {request.options.level!r} is not supported "
                        f"by {profiler.name!r}"
                    ),
                )
            else:
                effective_options, option_warnings = profiler.normalize_options(
                    request.options
                )
                raise_if_operation_cancelled()
                deadline = RequestDeadline(
                    started_at + effective_options.timeout_sec
                )
                deadline.remaining("evaluator command construction")

                from ..evaluation.adapters import create_adapter

                adapter = create_adapter(
                    request.binding,
                    device_string=device,
                    backend=self.backend,
                    operator_bundle_root=self.operator_bundle_root,
                )
                command = adapter.build_profile_command(
                    request,
                    effective_options,
                    artifact_dir,
                    device,
                )
                raise_if_operation_cancelled()
                deadline.remaining("vendor profiling")
                backend_result = profiler.profile_command(
                    command,
                    options=effective_options,
                    artifact_dir=artifact_dir,
                    device=device,
                    deadline=deadline,
                )
                raise_if_operation_cancelled()
        except OperationCancelled:
            raise
        except ProfileDeadlineExceeded as exc:
            backend_result = BackendProfileResult(
                status="failed",
                profiler=profiler.name,
                error=str(exc),
            )
        except Exception as exc:
            backend_result = BackendProfileResult(
                status="failed",
                profiler=profiler.name,
                error=f"unhandled profiler error: {exc}",
            )

        public_artifacts = []
        files = {}
        warnings = [*option_warnings, *backend_result.warnings]
        for index, source in enumerate(backend_result.artifacts, start=1):
            path = source.path.resolve()
            if not path.is_file() or not path.is_relative_to(artifact_dir):
                warnings.append(f"ignored missing or unsafe artifact: {source.path}")
                continue
            artifact_id = f"a{index:03d}"
            public_artifacts.append(
                ProfileArtifact(
                    id=artifact_id,
                    kind=source.kind,
                    format=source.format,
                    filename=path.name,
                    media_type=source.media_type,
                    size_bytes=path.stat().st_size,
                    download_url=f"/profile_artifacts/{profile_id}/{artifact_id}",
                )
            )
            files[artifact_id] = {
                "relative_path": str(path.relative_to(artifact_dir)),
                "filename": path.name,
                "media_type": source.media_type,
            }

        result = ProfileResult(
            profile_id=profile_id,
            status=backend_result.status,
            backend=self.backend,
            profiler=backend_result.profiler,
            device=device,
            evaluation_id=request.evaluation_id,
            workload_name=request.case_id,
            options=effective_options,
            hardware=backend_result.hardware or self.hardware,
            capabilities=backend_result.capabilities,
            summary=backend_result.summary,
            metrics=backend_result.metrics,
            artifacts=public_artifacts,
            warnings=warnings,
            error=backend_result.error,
        )
        manifest = {
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "files": files,
        }
        temporary = artifact_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(artifact_dir / "manifest.json")
        return result

    def resolve_artifact(
        self,
        profile_id: str,
        artifact_id: str,
    ) -> tuple[Path, str, str]:
        profile_dir = (self.root / profile_id).resolve()
        if not profile_dir.is_relative_to(self.root):
            raise ProfileArtifactNotFound(profile_id)
        manifest_path = profile_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ProfileArtifactNotFound(profile_id)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = manifest["files"][artifact_id]
            artifact_path = (profile_dir / entry["relative_path"]).resolve()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProfileArtifactNotFound(artifact_id) from exc
        if (
            not artifact_path.is_file()
            or not artifact_path.is_relative_to(profile_dir)
        ):
            raise ProfileArtifactNotFound(artifact_id)
        return (
            artifact_path,
            entry.get("media_type", "application/octet-stream"),
            entry.get("filename", artifact_path.name),
        )
