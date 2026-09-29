"""Immutable eval snapshots and backend-neutral profile tool implementations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

from kernelgen.data.ledger import Ledger
from kernelgen.data.profile_analysis import ProfileAnalysis, ProfileFinding
from kernelgen.data.tool_context import ToolContext
from kernelgen.tools.kernelgen_server_adapter import (
    EvaluationBundle,
    SERVER_API_VERSION,
)


EVALS_RELATIVE_DIR = Path(".kernelgen") / "evals"
PROFILES_RELATIVE_DIR = Path(".kernelgen") / "profiles"
ANALYSES_RELATIVE_DIR = Path(".kernelgen") / "profile-analysis"


def _relative_to_workspace(workspace: Path, path: Path) -> str:
    return str(path.resolve().relative_to(workspace.resolve()))


def _resolve_workspace_path(
    workspace: Path,
    raw_path: str,
    *,
    allow_absolute: bool = False,
) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and not allow_absolute:
        raise ValueError(f"absolute paths are not accepted from agents: {raw_path}")
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside the agent workspace: {raw_path}") from exc
    return resolved


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _catalog_identity(context: ToolContext) -> str:
    return context.catalog_name


def resolve_catalog_path(context: ToolContext) -> Path:
    from kernelgen.tools.kernelgen_server_adapter import (
        resolve_catalog_path as resolve_adapter_catalog_path,
    )

    return resolve_adapter_catalog_path(
        catalog_name=context.catalog_name,
    )


# Retained for callers in the legacy extraction test surface.
resolve_trace_set_path = resolve_catalog_path


def prepare_evaluation_bundle(kernel_path: Path, context: ToolContext) -> EvaluationBundle:
    """Load the exact objects shared by preflight, evaluation, and profiling."""
    if context.destination_passing_style:
        raise ValueError(
            "KernelGen Server schema v1 supports value-returning run() only; "
            "set destination_passing_style=false"
        )
    from kernelgen.tools.kernelgen_server_adapter import (
        prepare_evaluation_bundle as prepare_native_bundle,
    )

    return prepare_native_bundle(kernel_path, context)


def write_evaluation_snapshot(
    workspace: str | Path,
    round_num: int,
    bundle: EvaluationBundle,
    eval_result: dict,
    context: ToolContext,
) -> dict:
    """Persist an immutable, content-addressed snapshot for one measured round."""
    from kernelgen.tools.preflight import _canonical_sha256
    from kernelgen_client import EvaluatorBinding

    root = Path(workspace).resolve()
    destination = root / EVALS_RELATIVE_DIR / f"round-{round_num:04d}"
    if destination.exists():
        raise FileExistsError(f"evaluation snapshot already exists: {destination}")
    solution_data = bundle.solution.model_dump(mode="json")
    definition_data = bundle.definition.model_dump(mode="json")
    workload_data = [workload.model_dump(mode="json") for workload in bundle.workloads]
    workload_uuids = [workload["name"] for workload in workload_data]
    profile_workload_uuids = (
        list(bundle.profile_workload_uuids)
        if bundle.profile_workload_uuids is not None
        else list(workload_uuids)
    )
    unknown_profile_workloads = set(profile_workload_uuids) - set(workload_uuids)
    if unknown_profile_workloads:
        raise ValueError(
            "profile workloads are absent from the evaluation bundle: "
            f"{sorted(unknown_profile_workloads)}"
        )
    solution_sha = _canonical_sha256(solution_data)
    definition_sha = _canonical_sha256(definition_data)
    workload_hashes = [_canonical_sha256(workload) for workload in workload_data]
    binding = (bundle.binding or EvaluatorBinding(
        catalog_name=context.catalog_name,
        definition=bundle.definition.name,
    )).model_dump(mode="json", exclude_none=True)
    from kernelgen.tools.retest import retest_contract
    contract = retest_contract(bundle, context, eval_result.get("evaluation_service") or {})
    fingerprint = _canonical_sha256(
        {
            "eval_api_version": eval_result.get(
                "api_version",
                SERVER_API_VERSION,
            ),
            "profile_analysis_schema_version": "1.0",
            "solution_sha256": solution_sha,
            "definition_sha256": definition_sha,
            "workload_sha256": workload_hashes,
            "workload_uuids": workload_uuids,
            "workload_mode": "phased" if bundle.is_phased else "legacy",
            "profile_workload_uuids": profile_workload_uuids,
            "catalog_name": _catalog_identity(context),
            "binding": binding,
            "adapter_kind": bundle.adapter_kind,
            "benchmark_fingerprint": bundle.benchmark_fingerprint,
            "target_hardware": context.target_hardware,
            "implementation_language": context.implementation_language.value,
            "server_backend": eval_result.get("server_backend") or "unknown",
            "retest_contract": contract,
        }
    )
    identity = {
        "round_num": round_num,
        "evaluation_fingerprint": fingerprint,
        "solution_sha256": solution_sha,
        "definition_name": bundle.definition.name,
        "definition_sha256": definition_sha,
        "workload_uuids": workload_uuids,
        "workload_sha256": workload_hashes,
        "workload_mode": "phased" if bundle.is_phased else "legacy",
        "profile_workload_uuids": profile_workload_uuids,
        "catalog_name": _catalog_identity(context),
        "binding": binding,
        "adapter_kind": bundle.adapter_kind,
        "benchmark_fingerprint": bundle.benchmark_fingerprint,
        "target_hardware": context.target_hardware,
        "implementation_language": context.implementation_language.value,
        "server_backend": eval_result.get("server_backend") or "unknown",
        "retest_contract": contract,
        "result_sha256": _canonical_sha256(eval_result),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".round-{round_num:04d}-", dir=destination.parent))
    try:
        (temporary / "main.py").write_text(bundle.kernel_code, encoding="utf-8")
        for filename, payload in (
            ("result.json", eval_result),
            ("solution.json", solution_data),
            ("definition.json", definition_data),
            ("workloads.json", workload_data),
            ("identity.json", identity),
        ):
            (temporary / filename).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "evaluation_fingerprint": fingerprint,
        "solution_sha256": solution_sha,
        "snapshot_path": _relative_to_workspace(root, destination),
    }


def _load_snapshot(workspace: Path, round_num: int) -> dict:
    ledger = Ledger(workspace)
    record = ledger.get_round(round_num)
    if not record.solution.snapshot_path:
        raise ValueError(f"round {round_num} has no immutable evaluation snapshot")
    snapshot = _resolve_workspace_path(workspace, record.solution.snapshot_path)
    if not snapshot.is_dir():
        raise ValueError(f"evaluation snapshot is missing: {snapshot}")

    def read(name: str) -> Any:
        return json.loads((snapshot / name).read_text(encoding="utf-8"))

    return {
        "ledger": ledger,
        "record": record,
        "path": snapshot,
        "result": read("result.json"),
        "solution": read("solution.json"),
        "definition": read("definition.json"),
        "workloads": read("workloads.json"),
        "identity": read("identity.json"),
    }


def _profile_root(workspace: Path, round_num: int) -> Path:
    return workspace / PROFILES_RELATIVE_DIR / f"round-{round_num:04d}"


def _profile_workload_uuids(snapshot: dict) -> List[str]:
    """Return profile candidates, treating snapshots without phase data as legacy."""
    identity = snapshot["identity"]
    candidates = identity.get("profile_workload_uuids")
    if candidates is None:
        candidates = identity.get("workload_uuids", [])
    return list(dict.fromkeys(str(item) for item in candidates if str(item)))


def _workload_profile_root(workspace: Path, round_num: int, workload_uuid: str) -> Path:
    if not workload_uuid:
        raise ValueError("workload UUID must not be empty")

    # Adapter case IDs are opaque identifiers, not filesystem paths.  Pytest
    # adapters legitimately use IDs such as
    # ``benchmark/test_addmm_.py::test_addmm_::core::float16::0``.  Preserve
    # the historical, readable directory name for short filename-safe IDs,
    # and map every other valid opaque ID to a stable collision-resistant
    # directory.  The original ID remains authoritative in the manifest.
    if (
        workload_uuid not in {".", ".."}
        and Path(workload_uuid).name == workload_uuid
        and len(workload_uuid.encode("utf-8")) <= 128
    ):
        directory = workload_uuid
    else:
        directory = "workload-" + hashlib.sha256(
            workload_uuid.encode("utf-8")
        ).hexdigest()
    return _profile_root(workspace, round_num) / directory


def _cached_manifests(workspace: Path, round_num: int) -> List[dict]:
    manifests: List[dict] = []
    root = _profile_root(workspace, round_num)
    if not root.is_dir():
        return manifests
    for path in sorted(root.glob("*/*/manifest.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            result = value.get("result", {})
            manifests.append(
                {
                    "workload_uuid": result.get("workload_uuid"),
                    "profile_id": result.get("profile_id"),
                    "status": result.get("status"),
                    "level": (result.get("options") or {}).get("level"),
                    "manifest_path": _relative_to_workspace(workspace, path),
                }
            )
        except (OSError, ValueError, TypeError):
            continue
    return manifests


def get_profile_context(
    workspace: str | Path,
    context: ToolContext,
    round_num: int,
) -> dict:
    from kernelgen.tools.kernelgen_server_adapter import (
        get_service_status,
        require_target_context,
    )

    root = Path(workspace).resolve()
    snapshot = _load_snapshot(root, round_num)
    record = snapshot["record"]
    service = get_service_status(context.eval_server_url)
    require_target_context(context, service)
    return {
        "round_num": round_num,
        "evaluation_fingerprint": record.evaluation.fingerprint,
        "eval_status": record.evaluation.status,
        "solution_sha256": snapshot["identity"]["solution_sha256"],
        "candidate_path": _relative_to_workspace(root, snapshot["path"] / "main.py"),
        "per_workload": snapshot["result"].get("per_workload", []),
        "profile_workload_uuids": _profile_workload_uuids(snapshot),
        "experiment_plan": record.plan.model_dump(mode="json"),
        "evaluation_comparison": record.evaluation.comparison.model_dump(mode="json"),
        "service": service,
        "profile_state": record.profile.status,
        "cached_profiles": _cached_manifests(root, round_num),
    }


def get_workspace_profile_context(
    workspace: str | Path,
    round_num: int,
) -> dict:
    """Load workspace configuration and return trusted profile context."""
    from kernelgen.data.tool_context import load_tool_context

    root = Path(workspace).resolve()
    return get_profile_context(root, load_tool_context(root), round_num)


def _find_cached_profile(
    workspace: Path,
    round_num: int,
    workload_uuid: str,
    solution_sha256: str,
    workload_sha256: str,
    options: dict,
) -> dict | None:
    workload_root = _workload_profile_root(workspace, round_num, workload_uuid)
    if not workload_root.is_dir():
        return None
    for path in sorted(workload_root.glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        request = manifest.get("request", {})
        target = request.get("identity") or request.get("target", {})
        if (
            target.get("solution_sha256") == solution_sha256
            and target.get("workload_sha256") == workload_sha256
            and request.get("options") == options
        ):
            result = manifest.get("result")
            if not isinstance(result, dict) or result.get("status") != "completed":
                continue
            result["manifest_path"] = str(path)
            result["cached"] = True
            return result
    return None


def _compact_profile_result(workspace: Path, result: dict) -> dict:
    artifacts = []
    for artifact in result.get("artifacts", []):
        local_path = artifact.get("local_path")
        if local_path:
            local_path = _relative_to_workspace(
                workspace,
                _resolve_workspace_path(workspace, local_path, allow_absolute=True),
            )
        artifacts.append(
            {
                "id": artifact.get("id"),
                "kind": artifact.get("kind"),
                "format": artifact.get("format"),
                "local_path": local_path,
                "download_error": artifact.get("download_error"),
            }
        )
    manifest_path = result.get("manifest_path", "")
    if manifest_path:
        manifest_path = _relative_to_workspace(
            workspace,
            _resolve_workspace_path(workspace, manifest_path, allow_absolute=True),
        )
    return {
        "cached": bool(result.get("cached")),
        "status": result.get("status"),
        "profile_id": result.get("profile_id"),
        "workload_uuid": result.get("workload_uuid"),
        "backend": result.get("backend"),
        "profiler": result.get("profiler"),
        "capabilities": result.get("capabilities", []),
        "summary": result.get("summary", {}),
        "metrics": result.get("metrics", {}),
        "artifacts": artifacts,
        "manifest_path": manifest_path,
        "warnings": result.get("warnings", []),
        "error": result.get("error"),
    }


def profile_workloads(
    workspace: str | Path,
    context: ToolContext,
    round_num: int,
    workload_uuids: Iterable[str],
    *,
    level: str = "metrics",
    backend_options: Dict[str, Dict[str, Any]] | None = None,
    run_control: Any | None = None,
) -> dict:
    """Profile any number of exact workloads from an immutable eval snapshot."""
    from urllib.parse import urljoin
    from urllib.request import urlopen

    from kernelgen_client import EvaluatorBinding, Implementation, Workload
    from kernelgen_client.http import profile
    from kernelgen_client.profiling import (
        ProfileOptions,
        ProfileRequest,
    )
    from kernelgen.tools.kernelgen_server_adapter import (
        get_service_status,
        require_target_context,
        resolve_server_url,
        tracked_server_operation,
    )

    if level not in {"metrics", "source", "instruction"}:
        raise ValueError("level must be metrics, source, or instruction")
    requested = list(dict.fromkeys(str(item) for item in workload_uuids if str(item)))
    if not requested:
        raise ValueError("workload_uuids must contain at least one UUID")

    root = Path(workspace).resolve()
    snapshot = _load_snapshot(root, round_num)
    record = snapshot["record"]
    if record.evaluation.status != "PASSED":
        raise ValueError(f"round {round_num} is {record.evaluation.status}, not PASSED")
    if not record.profile.required or record.profile.status not in {"pending", "collecting"}:
        raise ValueError(f"round {round_num} profile state is {record.profile.status!r}")

    solution = Implementation.model_validate(snapshot["solution"])
    workload_map = {
        item["name"]: Workload.model_validate(item)
        for item in snapshot["workloads"]
    }
    unknown = [item for item in requested if item not in workload_map]
    if unknown:
        raise ValueError(f"unknown workload UUIDs for round {round_num}: {unknown}")
    profile_candidates = set(_profile_workload_uuids(snapshot))
    ineligible = [item for item in requested if item not in profile_candidates]
    if ineligible:
        raise ValueError(
            "workload UUIDs are not timing/profile candidates for round "
            f"{round_num}: {ineligible}"
        )
    for workload_uuid in requested:
        _workload_profile_root(root, round_num, workload_uuid)

    options_model = ProfileOptions(level=level, backend_options=backend_options or {})
    options_data = options_model.model_dump(mode="json")
    # Older installed-Catalog snapshots predate explicit evaluator bindings.
    binding = EvaluatorBinding.model_validate(snapshot["identity"].get("binding") or {
        "catalog_name": snapshot["identity"]["catalog_name"],
        "definition": snapshot["identity"]["definition_name"],
    })
    service = get_service_status(context.eval_server_url)
    require_target_context(context, service)
    snapshot["ledger"].mark_profile_collecting(round_num)

    backend = service.get("backend") or snapshot["identity"].get("server_backend")
    server_url = resolve_server_url(context.eval_server_url)
    workload_hashes = dict(
        zip(snapshot["identity"]["workload_uuids"], snapshot["identity"]["workload_sha256"])
    )
    profiles = []
    for workload_uuid in requested:
        cached = _find_cached_profile(
            root,
            round_num,
            workload_uuid,
            snapshot["identity"]["solution_sha256"],
            workload_hashes[workload_uuid],
            options_data,
        )
        if cached is not None:
            profiles.append(_compact_profile_result(root, cached))
            continue

        request_id = uuid.uuid4().hex
        output_dir = _workload_profile_root(root, round_num, workload_uuid) / request_id
        request = ProfileRequest(
            evaluation_id=(
                f"{record.evaluation.fingerprint}:round-{round_num}"
            ),
            binding=binding,
            implementation=solution,
            benchmark_fingerprint=snapshot["identity"][
                "benchmark_fingerprint"
            ],
            case_id=workload_uuid,
            expected_backend=backend,
            options=options_model,
        )
        with tracked_server_operation(
            run_control,
            service,
            server_url,
            "profile",
        ) as operation_id:
            response = profile(request, server_url, operation_id=operation_id)
        result = response.model_dump(mode="json")
        result["workload_uuid"] = result.pop("workload_name")
        output_dir.mkdir(parents=True, exist_ok=True)
        local_artifacts = []
        for artifact in result.get("artifacts", []):
            filename = Path(artifact["filename"]).name
            local_path = output_dir / filename
            try:
                with urlopen(
                    urljoin(server_url.rstrip("/") + "/", artifact["download_url"]),
                    timeout=options_model.timeout_sec,
                ) as remote, local_path.open("wb") as local:
                    shutil.copyfileobj(remote, local)
                download_error = None
            except Exception as exc:  # noqa: BLE001
                download_error = str(exc)
            local_artifacts.append(
                {
                    **artifact,
                    "local_path": str(local_path) if local_path.is_file() else None,
                    "download_error": download_error,
                }
            )
        result["artifacts"] = local_artifacts
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "request": {
                        **request.model_dump(mode="json"),
                        "identity": {
                            "solution_sha256": snapshot["identity"][
                                "solution_sha256"
                            ],
                            "workload_sha256": workload_hashes[workload_uuid],
                        },
                    },
                    "result": result,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result["manifest_path"] = str(manifest)
        profiles.append(_compact_profile_result(root, result))

    statuses = {item.get("status") for item in profiles}
    if statuses == {"completed"}:
        overall = "completed"
    elif "completed" in statuses:
        overall = "partial"
    elif "failed" in statuses:
        overall = "failed"
    elif "unsupported" in statuses:
        overall = "unsupported"
    else:
        overall = "failed"
    return {
        "round_num": round_num,
        "evaluation_fingerprint": record.evaluation.fingerprint,
        "status": overall,
        "backend": backend,
        "service_profile": service.get("profile", {}),
        "profiles": profiles,
    }


def profile_workspace_workloads(
    workspace: str | Path,
    round_num: int,
    workload_uuids: Iterable[str],
    *,
    level: str = "metrics",
    backend_options: Dict[str, Dict[str, Any]] | None = None,
) -> dict:
    """Load workspace configuration and profile selected immutable workloads."""
    from kernelgen.data.tool_context import load_tool_context
    from kernelgen.framework.run_control import (
        RunState,
        WorkspaceRunControl,
        cooperative_cancel_result,
    )
    from kernelgen_client.http import OperationCancelledError

    root = Path(workspace).resolve()
    run_control = WorkspaceRunControl(root, source="mcp:profile_workloads")
    cancelled = cooperative_cancel_result(
        run_control,
        stage="BEFORE_PROFILE",
    )
    if cancelled is not None:
        return cancelled
    run_control.update_progress(
        state=RunState.RUNNING,
        stage="PROFILING",
        message=f"Profiling round {round_num}",
    )
    run_control.record_event(
        "PROFILE_STARTED",
        message=f"Profiling round {round_num}",
        stage="PROFILING",
        data={"round_num": round_num, "level": level},
    )
    try:
        result = profile_workloads(
            root,
            load_tool_context(root),
            round_num,
            workload_uuids,
            level=level,
            backend_options=backend_options,
            run_control=run_control,
        )
    except OperationCancelledError as exc:
        cancelled = cooperative_cancel_result(
            run_control,
            stage="DURING_PROFILE",
        ) or {
            "status": "RUN_CANCELLED",
            "cancel_requested": True,
            "reason_code": "server_operation_cancelled",
            "reason": str(exc),
            "stage": "DURING_PROFILE",
            "instruction": (
                "STOP NOW. Return the final report using only results "
                "already persisted."
            ),
        }
        run_control.record_event(
            "PROFILE_CANCELLED",
            message=cancelled["reason"],
            stage="PROFILING",
            level="WARNING",
            data={"round_num": round_num},
        )
        return cancelled
    except Exception as exc:
        run_control.record_event(
            "PROFILE_FAILED",
            message=str(exc),
            stage="PROFILING",
            level="ERROR",
            visibility="DEBUG",
            data={"round_num": round_num},
        )
        raise
    run_control.update_progress(
        state=RunState.RUNNING,
        stage="PROFILE_ANALYSIS",
        message=f"Profile {result.get('status', 'UNKNOWN')}",
    )
    run_control.record_event(
        "PROFILE_COMPLETED",
        message=f"Profile {result.get('status', 'UNKNOWN')}",
        stage="PROFILING",
        level=(
            "INFO" if result.get("status") == "completed" else "WARNING"
        ),
        data={
            "round_num": round_num,
            "status": result.get("status"),
            "backend": result.get("backend"),
            "profile_count": len(result.get("profiles") or []),
        },
    )
    return result


def _all_manifest_evidence(workspace: Path, round_num: int) -> tuple[set[str], set[Path]]:
    profile_ids: set[str] = set()
    artifact_paths: set[Path] = set()
    root = _profile_root(workspace, round_num).resolve()
    if not root.is_dir():
        return profile_ids, artifact_paths
    for manifest_path in root.glob("*/*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        result = manifest.get("result", {})
        if result.get("profile_id"):
            profile_ids.add(result["profile_id"])
        for artifact in result.get("artifacts", []):
            if artifact.get("local_path"):
                try:
                    path = _resolve_workspace_path(
                        workspace,
                        artifact["local_path"],
                        allow_absolute=True,
                    )
                except ValueError:
                    continue
                if path.is_file() and path.is_relative_to(root):
                    artifact_paths.add(path)
    return profile_ids, artifact_paths


def validate_profile_query_findings(
    workspace: str | Path,
    round_num: int,
    findings: list[ProfileFinding],
):
    """Validate advisory post-profile query findings against local artifacts."""

    from kernelgen.knowledge.models import ProfileFindingContext

    if not findings:
        raise ValueError("post_profile draft_findings cannot be empty")
    root = Path(workspace).resolve()
    snapshot = _load_snapshot(root, round_num)
    record = snapshot["record"]
    if record.evaluation.status != "PASSED":
        raise ValueError(
            f"round {round_num} is {record.evaluation.status}, not PASSED"
        )
    if (
        not record.profile.required
        or record.profile.status not in {"pending", "collecting"}
    ):
        raise ValueError(
            f"round {round_num} profile state is {record.profile.status!r}"
        )
    known_uuids = set(snapshot["identity"]["workload_uuids"])
    _, artifact_paths = _all_manifest_evidence(root, round_num)
    artifact_workloads: dict[Path, str] = {}
    profile_root = _profile_root(root, round_num).resolve()
    for manifest_path in profile_root.glob("*/*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        relative = manifest_path.relative_to(profile_root)
        path_workload = relative.parts[0] if relative.parts else ""
        result = manifest.get("result", {})
        workload_uuid = str(result.get("workload_uuid") or path_workload)
        for artifact in result.get("artifacts", []):
            raw_path = artifact.get("local_path")
            if not raw_path:
                continue
            try:
                artifact_path = _resolve_workspace_path(
                    root,
                    raw_path,
                    allow_absolute=True,
                )
            except ValueError:
                continue
            if artifact_path in artifact_paths:
                artifact_workloads[artifact_path] = workload_uuid

    contexts = []
    for finding in findings:
        finding_workloads = set(finding.workload_uuids)
        unknown = finding_workloads - known_uuids
        if unknown:
            raise ValueError(
                f"draft finding {finding.label!r} references unknown workloads: "
                f"{sorted(unknown)}"
            )
        evidence_workloads = set()
        for evidence in finding.evidence:
            artifact_path = _resolve_workspace_path(
                root,
                evidence.artifact_path,
            )
            if artifact_path not in artifact_paths:
                raise ValueError(
                    "draft finding evidence is not a downloaded artifact for "
                    f"round {round_num}: {evidence.artifact_path}"
                )
            workload_uuid = artifact_workloads.get(artifact_path)
            if workload_uuid:
                evidence_workloads.add(workload_uuid)
        if not finding_workloads.issubset(evidence_workloads):
            raise ValueError(
                f"draft finding {finding.label!r} lacks artifact evidence for "
                f"workloads: {sorted(finding_workloads - evidence_workloads)}"
            )
        if not evidence_workloads.issubset(finding_workloads):
            raise ValueError(
                f"draft finding {finding.label!r} uses evidence from unclaimed "
                f"workloads: {sorted(evidence_workloads - finding_workloads)}"
            )
        contexts.append(
            ProfileFindingContext(
                category=finding.category,
                label=finding.label,
                confidence=finding.confidence,
                workload_uuids=finding.workload_uuids,
            )
        )
    return contexts


def record_profile_analysis(
    workspace: str | Path,
    round_num: int,
    analysis_data: dict,
) -> dict:
    """Validate evidence, atomically persist analysis, and clear the profile gate."""
    root = Path(workspace).resolve()
    snapshot = _load_snapshot(root, round_num)
    record = snapshot["record"]
    if not record.profile.required or record.profile.status not in {"pending", "collecting"}:
        raise ValueError(f"round {round_num} profile state is {record.profile.status!r}")
    analysis = ProfileAnalysis.model_validate(analysis_data)
    if analysis.round_num != round_num:
        raise ValueError("analysis round_num does not match the tool argument")
    if analysis.evaluation_fingerprint != record.evaluation.fingerprint:
        raise ValueError("analysis evaluation_fingerprint does not match the eval snapshot")
    if analysis.solution_sha256 != snapshot["identity"]["solution_sha256"]:
        raise ValueError("analysis solution_sha256 does not match the eval snapshot")
    expected_backend = snapshot["identity"].get("server_backend") or "unknown"
    if expected_backend != "unknown" and analysis.backend != expected_backend:
        raise ValueError(
            f"analysis backend {analysis.backend!r} does not match eval backend {expected_backend!r}"
        )

    known_uuids = set(snapshot["identity"]["workload_uuids"])
    used_uuids = {workload.uuid for workload in analysis.profiled_workloads}
    if not used_uuids.issubset(known_uuids):
        raise ValueError(f"analysis references unknown workload UUIDs: {sorted(used_uuids - known_uuids)}")

    authoritative_workloads = {
        item.get("uuid"): item
        for item in snapshot["result"].get("per_workload", [])
        if item.get("uuid")
    }
    for workload in analysis.profiled_workloads:
        authoritative = authoritative_workloads.get(workload.uuid, {})
        workload.axes = dict(authoritative.get("axes") or {})
        workload.eval_speedup = authoritative.get("speedup")
        workload.eval_latency_ms = authoritative.get("latency_ms")
        workload.eval_reference_latency_ms = authoritative.get("reference_latency_ms")

    profile_ids, artifact_paths = _all_manifest_evidence(root, round_num)
    referenced_profile_ids = {
        profile_id
        for workload in analysis.profiled_workloads
        for profile_id in workload.profile_ids
    }
    if not referenced_profile_ids.issubset(profile_ids):
        raise ValueError(
            f"analysis references unknown profile IDs: {sorted(referenced_profile_ids - profile_ids)}"
        )
    for workload in analysis.profiled_workloads:
        workload_manifest_profile_ids: set[str] = set()
        for manifest_raw in workload.manifest_paths:
            manifest_path = _resolve_workspace_path(root, manifest_raw)
            if not manifest_path.is_file() or not manifest_path.is_relative_to(
                _profile_root(root, round_num).resolve()
            ):
                raise ValueError(f"invalid manifest path for round {round_num}: {manifest_raw}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_result = manifest.get("result", {})
            manifest_uuid = manifest_result.get("workload_uuid")
            if manifest_uuid and manifest_uuid != workload.uuid:
                raise ValueError(
                    f"manifest {manifest_raw} belongs to workload {manifest_uuid!r}, not {workload.uuid!r}"
                )
            if manifest_result.get("profile_id"):
                workload_manifest_profile_ids.add(manifest_result["profile_id"])
        if not set(workload.profile_ids).issubset(workload_manifest_profile_ids):
            raise ValueError(
                f"workload {workload.uuid!r} profile IDs do not match its manifests"
            )
    for finding in analysis.findings:
        if not set(finding.workload_uuids).issubset(known_uuids):
            raise ValueError(f"finding {finding.label!r} references an unknown workload")
        if analysis.status == "completed" and not set(finding.workload_uuids).issubset(
            used_uuids
        ):
            raise ValueError(f"finding {finding.label!r} references an unprofiled workload")
        for evidence in finding.evidence:
            artifact_path = _resolve_workspace_path(root, evidence.artifact_path)
            if artifact_path not in artifact_paths:
                raise ValueError(
                    f"evidence path is not a downloaded artifact for round {round_num}: {evidence.artifact_path}"
                )

    analysis_path = root / ANALYSES_RELATIVE_DIR / f"round-{round_num:04d}.json"
    _atomic_json(analysis_path, analysis.model_dump(mode="json"))
    relative_analysis_path = _relative_to_workspace(root, analysis_path)
    snapshot["ledger"].attach_profile_analysis(
        round_num,
        status=analysis.status,
        analysis_path=relative_analysis_path,
        summary=analysis.compact_summary(),
    )
    result = {
        "recorded": True,
        "round_num": round_num,
        "status": analysis.status,
        "analysis_path": relative_analysis_path,
        "summary": analysis.compact_summary(),
    }
    from kernelgen.framework.run_control import RunState, WorkspaceRunControl

    run_control = WorkspaceRunControl(
        root,
        source="mcp:record_profile_analysis",
    )
    run_control.update_progress(
        state=RunState.RUNNING,
        stage="FINALIZING_ROUND",
        message=f"Profile analysis {analysis.status}",
    )
    run_control.record_event(
        "PROFILE_ANALYSIS_RECORDED",
        message=analysis.compact_summary(),
        stage="PROFILE_ANALYSIS",
        data={"round_num": round_num, "status": analysis.status},
    )
    return result
