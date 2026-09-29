"""Deterministic context construction from orchestrator-owned inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from kernelgen.knowledge.config import KnowledgeMode
from kernelgen.knowledge.contracts.runtime import WorkspaceKnowledgeState
from kernelgen.knowledge.models import (
    OperatorSignature,
    QueryContext,
    SourceReference,
    TargetContext,
    ProfileFindingContext,
    SoftwareContext,
)
from kernelgen.knowledge.layout import KnowledgeLayout
from kernelgen.knowledge.taxonomy import OperatorTaxonomy


class KnowledgeWorkspaceMaterializer:
    """Write one workspace's immutable knowledge context."""

    def __init__(
        self,
        *,
        catalog_root: Path,
        mode: KnowledgeMode,
        run_id: str,
        operator_signature: OperatorSignature,
        target_context: TargetContext,
        start_mode: str = "fresh",
        parent_solution_ref: str = "",
        derived_root: Path | None = None,
        index_rebuild_enabled: bool = True,
    ):
        self.catalog_root = Path(catalog_root)
        self.mode = mode
        self.run_id = run_id
        self.operator_signature = operator_signature
        self.target_context = target_context
        self.start_mode = start_mode
        self.parent_solution_ref = parent_solution_ref
        self.derived_root = (
            Path(derived_root) if derived_root is not None else None
        )
        self.index_rebuild_enabled = index_rebuild_enabled

    def materialize(self, workspace: Path) -> None:
        workspace = Path(workspace)
        legacy_copy = workspace / "kb"
        if (
            legacy_copy.exists()
            and legacy_copy.resolve() != self.catalog_root.resolve()
        ):
            shutil.rmtree(legacy_copy)
        layout = KnowledgeLayout(workspace)
        layout.context_dir.mkdir(parents=True, exist_ok=True)
        layout.operator_signature.write_text(
            self.operator_signature.model_dump_json(
                indent=2,
                exclude_none=True,
            ),
            encoding="utf-8",
        )
        layout.target_context.write_text(
            self.target_context.model_dump_json(indent=2),
            encoding="utf-8",
        )
        state = WorkspaceKnowledgeState(
            catalog_ref=str(self.catalog_root.resolve()),
            mode=self.mode.value,
            derived_ref=(
                str(self.derived_root.resolve())
                if self.derived_root is not None
                else ""
            ),
            index_rebuild_enabled=self.index_rebuild_enabled,
            run_id=self.run_id,
            workspace_id=self._workspace_id(workspace),
            operator_signature_ref=str(
                layout.operator_signature.relative_to(workspace)
            ),
            target_context_ref=str(layout.target_context.relative_to(workspace)),
            start_mode=self.start_mode,
            parent_solution_ref=self.parent_solution_ref,
        )
        layout.state.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def _workspace_id(self, workspace: Path) -> str:
        """Return a run-unique, Observation-ID-safe workspace identity."""

        try:
            relative = workspace.resolve().relative_to(
                self.catalog_root.resolve().parent
            )
            parts = relative.parts
        except ValueError:
            parts = (workspace.parent.name, workspace.name)
        normalized = ".".join(
            "".join(
                character if character.isalnum() or character in "._-" else "-"
                for character in part
            ).strip("-")
            or "unknown"
            for part in parts
        )
        return normalized or "unknown"


def build_operator_signature(
    definition: Any,
    *,
    catalog_root: Path | None = None,
    taxonomy: OperatorTaxonomy | None = None,
) -> OperatorSignature:
    payload = (
        definition.model_dump(mode="json")
        if hasattr(definition, "model_dump")
        else dict(definition)
    )
    name = str(payload.get("name") or "")
    op_type = str(payload.get("op_type") or "")
    definition_id = name or op_type or "unknown"
    if taxonomy is None and catalog_root is not None:
        taxonomy = OperatorTaxonomy.from_catalog(catalog_root)
    mapping = taxonomy.mapping_for(definition_id) if taxonomy is not None else None
    if mapping is None:
        motifs = _legacy_motifs(payload, name=name, op_type=op_type)
        dataflow = _normalized_strings(payload.get("dataflow"))
        mapped_dtypes: list[str] = []
        mapped_layouts: list[str] = []
        mapped_features: dict[str, Any] = {}
    else:
        op_type = mapping.family
        motifs = list(mapping.motifs)
        dataflow = list(mapping.dataflow)
        mapped_dtypes = list(mapping.dtypes)
        mapped_layouts = list(mapping.layouts)
        mapped_features = dict(mapping.workload_features)

    observed_dtypes = {
        _normalize_dtype(spec.get("dtype"))
        for group in ("inputs", "outputs")
        for spec in _tensor_specs(payload.get(group))
        if spec.get("dtype")
    }
    dtypes = sorted(set(mapped_dtypes) | observed_dtypes)
    workload_features = dict(mapped_features)
    workload_features.update({
        key: value.get("value")
        for key, value in (payload.get("axes") or {}).items()
        if isinstance(value, dict) and value.get("value") is not None
    })
    layouts = [*mapped_layouts, *_normalized_strings(payload.get("layouts"))]
    for group in ("inputs", "outputs"):
        for spec in _tensor_specs(payload.get(group)):
            if spec.get("layout"):
                layouts.extend(_normalized_strings([spec["layout"]]))
    layouts = sorted(set(layouts))
    numerics = dict(payload.get("numerics") or {})
    if numerics.get("exact") is False:
        numerics.pop("exact")
    return OperatorSignature(
        definition_id=definition_id,
        definition_name=name,
        op_type=op_type,
        motifs=motifs,
        dataflow=dataflow,
        dtypes=dtypes,
        layouts=layouts,
        workload_features=workload_features,
        required_capabilities=_normalized_strings(payload.get("capabilities")),
        numerics=numerics,
    )


def _legacy_motifs(
    payload: dict[str, Any],
    *,
    name: str,
    op_type: str,
) -> list[str]:
    reference = str(payload.get("reference") or "").lower()
    combined = f"{name} {op_type} {reference}"
    motifs = []
    for motif, tokens in (
        ("reduction", ("sum(", "max(", "min(", "reduce", "norm", "softmax")),
        ("elementwise", ("relu", "gelu", "sigmoid", "where(", "+", "*")),
        ("matrix_multiply", ("matmul", "mm(", "@")),
        ("normalization", ("norm", "mean(", "variance", "var(")),
        ("attention", ("attention", "softmax", "qk", "query", "key", "value")),
        ("sort", ("sort(", "argsort", "topk", "top_k")),
        ("broadcast", ("broadcast", "broadcast_to", "expand_as", "expand(")),
        ("conversion", ("transpose", "permute", "concat", "cat(", "split(")),
    ):
        if any(token in combined for token in tokens):
            motifs.append(motif)
    return motifs


def _tensor_specs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def build_target_context(
    *,
    target_hardware: str,
    implementation_language: str,
    service_status: dict[str, Any] | None = None,
) -> TargetContext:
    from kernelgen.data.target_context import (
        build_target_context as build_runtime_target_context,
    )

    runtime = build_runtime_target_context(
        target_hardware=target_hardware,
        implementation_language=implementation_language,
        service_status=service_status or {},
    )
    return TargetContext(
        backend=runtime.backend,
        vendor=runtime.vendor,
        architecture=runtime.architecture,
        device=runtime.device,
        capabilities=runtime.capabilities,
        software=SoftwareContext(
            **runtime.software.model_dump(mode="python")
        ),
        metadata=(
            runtime.metadata.model_dump(mode="python")
            if runtime.metadata is not None
            else None
        ),
        source="eval_service",
    )


def normalize_operator_signature(
    signature: OperatorSignature,
) -> OperatorSignature:
    """Convert legacy hash identity to the current human definition identity."""
    return signature.model_copy(
        update={"definition_id": signature.definition_name}
    )


def build_query_context(
    workspace: Path,
    *,
    phase: str,
    task: str,
    question: str,
    round_num: int | None = None,
    max_results: int = 12,
    draft_findings: list[ProfileFindingContext] | None = None,
) -> QueryContext:
    layout = KnowledgeLayout(Path(workspace))
    signature = OperatorSignature.model_validate_json(
        layout.operator_signature.read_text(encoding="utf-8")
    )
    target = TargetContext.model_validate_json(
        layout.target_context.read_text(encoding="utf-8")
    )
    signature = normalize_operator_signature(signature)
    findings = list(draft_findings or [])
    errors = []
    if round_num is not None:
        from kernelgen.data.ledger import Ledger

        try:
            record = Ledger(workspace).get_round(round_num)
        except ValueError:
            record = None
        if record is not None:
            if (
                phase == "post_profile"
                and not findings
                and record.profile.analysis_path
            ):
                analysis_path = Path(workspace) / record.profile.analysis_path
                if analysis_path.is_file():
                    try:
                        payload = json.loads(
                            analysis_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError):
                        payload = {}
                    findings = [
                        ProfileFindingContext(
                            category=str(item.get("category") or "unknown"),
                            label=str(item.get("label") or ""),
                            confidence=str(item.get("confidence") or "low"),
                            workload_uuids=list(item.get("workload_uuids") or []),
                        )
                        for item in payload.get("findings", [])
                        if item.get("label")
                    ]
            if phase == "post_error" and record.evaluation.status != "PASSED":
                errors.append(
                    f"{record.evaluation.status}: {record.evaluation.log}".strip()
                )
    return QueryContext(
        phase=phase,
        task=task,
        round_num=round_num,
        question=question,
        operator_signature=signature,
        target_context=target,
        findings=findings,
        errors=errors,
        max_results=max_results,
    )


def workspace_provenance(
    workspace: Path,
) -> tuple[list[SourceReference], list[str]]:
    """Return only Sources and Concepts actually retrieved in this workspace."""
    layout = KnowledgeLayout(Path(workspace))
    sources: dict[tuple[str, str, str], SourceReference] = {}
    concept_ids: set[str] = set()
    if not layout.retrieval_log.is_file():
        return [], []
    for line in layout.retrieval_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if record.get("status") != "success":
            continue
        if record.get("operation") == "get_source":
            for raw in record.get("returned_sources") or []:
                identity, separator, locator = str(raw).partition("::")
                resource, revision_separator, revision = identity.rpartition("@")
                if not revision_separator:
                    resource = identity
                    revision = ""
                if separator and resource.startswith("source:") and locator:
                    key = (resource, revision, locator)
                    sources[key] = SourceReference(
                        resource=resource,
                        revision=revision,
                        locator=locator,
                    )
        if record.get("operation") == "get_knowledge":
            for raw in record.get("returned_refs") or []:
                concept_id = str(raw)
                legacy_id, separator, revision = concept_id.rpartition("@")
                if separator and revision.isdigit():
                    concept_id = legacy_id
                if concept_id.startswith("kg:"):
                    concept_ids.add(concept_id)
    return (
        [sources[key] for key in sorted(sources)],
        sorted(concept_ids),
    )


def _normalize_backend(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    return {
        "npu": "ascend",
        "cann": "ascend",
        "ascend_npu": "ascend",
        "gpu": "cuda",
        "nvidia": "cuda",
    }.get(normalized, normalized)


def normalize_device(value: Any) -> str:
    """Return a stable family name for known service/legacy device aliases."""

    from kernelgen.data.target_context import normalize_device as normalize

    return normalize(value)


def _normalized_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted(
        {
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        }
    )


def _normalize_dtype(value: Any) -> str:
    raw = getattr(value, "value", value)
    normalized = str(raw or "").strip().lower()
    if "." in normalized:
        prefix, suffix = normalized.rsplit(".", 1)
        if prefix in {"dtype", "torch", "numpy", "np"}:
            normalized = suffix
    return normalized
