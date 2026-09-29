"""Immutable run-level snapshots of Server Definitions and Workloads."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernelgen.data._atomic import atomic_write_json
from kernelgen.data.catalog import (
    load_catalog_manifest,
    resolve_builtin_catalog_path,
)


EVALUATION_SNAPSHOT_RELATIVE_PATH = (
    Path(".kernelgen") / "evaluation-contract.json"
)


class CatalogEvaluationSnapshot(BaseModel):
    """One catalog operator frozen before an optimization run starts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    catalog_name: str = Field(min_length=1)
    catalog_api_version: str = Field(min_length=1)
    catalog_generated_by: str = ""
    definition_name: str = Field(min_length=1)
    group: str = ""
    definition: dict[str, Any]
    correctness_workloads: list[dict[str, Any]]
    timing_workloads: list[dict[str, Any]]
    bundle_id: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    benchmark_fingerprint: str = ""
    evaluator_kind: Literal["native", "flaggems"] = "native"

    @model_validator(mode="after")
    def validate_native_contract(self) -> "CatalogEvaluationSnapshot":
        from kernelgen_client import Definition, Workload

        if self.bundle_id is not None and not self.benchmark_fingerprint:
            raise ValueError("bundle snapshot requires the verified benchmark fingerprint")
        if self.evaluator_kind == "flaggems" and not self.benchmark_fingerprint:
            raise ValueError("Gems snapshot requires an inspected Catalog binding")

        definition = Definition.model_validate(self.definition)
        has_input_factory = any(
            isinstance(node, ast.FunctionDef) and node.name == "gen_inputs"
            for node in ast.parse(definition.reference or "").body
        )
        if definition.name != self.definition_name:
            raise ValueError(
                "snapshot definition_name differs from Definition.name: "
                f"{self.definition_name!r} != {definition.name!r}"
            )
        names = []
        parameter_names = {parameter.name for parameter in definition.parameters}
        required_names = {
            parameter.name
            for parameter in definition.parameters
            if parameter.required and parameter.kind.value != "var_positional"
        }
        for raw in self.correctness_workloads + self.timing_workloads:
            workload = Workload.model_validate(raw)
            provided_names = set(workload.inputs)
            unknown_names = provided_names - parameter_names
            missing_names = required_names - provided_names
            # With gen_inputs these are an oracle-owned recipe, not invocation
            # kwargs. KGS validates the generated arguments on the target.
            if (unknown_names or missing_names) and not has_input_factory:
                raise ValueError(
                    f"snapshot workload {workload.name!r} inputs violate Definition "
                    f"{definition.name!r}: missing={sorted(missing_names)}, "
                    f"unknown={sorted(unknown_names)}"
                )
            names.append(workload.name)
        if len(names) != len(set(names)):
            raise ValueError("snapshot workload names must be globally unique")
        if not names and self.evaluator_kind == "native":
            raise ValueError("snapshot must contain at least one workload")
        return self

    @classmethod
    def from_server_contract(cls, contract, benchmark_fingerprint: str):
        """Freeze only target-owned facts; a fingerprint marks inspected bindings."""
        if not benchmark_fingerprint:
            raise ValueError("server snapshot requires a benchmark fingerprint")
        return cls(
            catalog_name=contract.binding.catalog_name or "uploaded-" + contract.binding.bundle_id.removeprefix("sha256:"),
            catalog_api_version=contract.catalog_api_version,
            definition_name=contract.binding.definition,
            definition=contract.definition.model_dump(mode="json", exclude_unset=True),
            correctness_workloads=[w.model_dump(mode="json", exclude_unset=True) for w in contract.correctness_workloads],
            timing_workloads=[w.model_dump(mode="json", exclude_unset=True) for w in contract.timing_workloads],
            bundle_id=contract.binding.bundle_id,
            benchmark_fingerprint=benchmark_fingerprint,
            evaluator_kind=contract.kind,
        )

    def native_operator(self) -> tuple[Any, list[Any], list[Any]]:
        from kernelgen_client import Definition, Workload

        definition = Definition.model_validate(self.definition)
        correctness = [
            Workload.model_validate(item) for item in self.correctness_workloads
        ]
        timing = [Workload.model_validate(item) for item in self.timing_workloads]
        return definition, correctness, timing


def load_catalog_evaluation_snapshot(
    path: str | Path,
) -> CatalogEvaluationSnapshot:
    snapshot_path = Path(path).expanduser().resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"evaluation contract snapshot is missing: {snapshot_path}"
        )
    return CatalogEvaluationSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )


def build_catalog_evaluation_snapshot(
    catalog_name: str,
    definition_name: str,
    *,
    catalog_path: str | Path | None = None,
) -> CatalogEvaluationSnapshot:
    from kernelgen_client import Catalog

    resolved_catalog_path = (
        Path(catalog_path).resolve()
        if catalog_path is not None
        else resolve_builtin_catalog_path(catalog_name)
    )
    manifest = load_catalog_manifest(catalog_name, path=resolved_catalog_path)
    catalog = Catalog(resolved_catalog_path)
    operator = catalog.load(definition_name)
    manifest_entry = next(
        (
            item
            for item in manifest.get("operators", [])
            if item.get("name") == definition_name
        ),
        None,
    )
    if manifest_entry is None:
        raise ValueError(
            f"definition {definition_name!r} is absent from catalog manifest"
        )
    return CatalogEvaluationSnapshot(
        catalog_name=catalog_name,
        catalog_api_version=str(manifest.get("api_version") or ""),
        catalog_generated_by=str(manifest.get("generated_by") or ""),
        definition_name=definition_name,
        group=str(manifest_entry.get("group") or ""),
        definition=operator.definition.model_dump(mode="json", exclude_unset=True),
        correctness_workloads=[
            workload.model_dump(mode="json", exclude_unset=True)
            for workload in operator.correctness_workloads
        ],
        timing_workloads=[
            workload.model_dump(mode="json", exclude_unset=True)
            for workload in operator.timing_workloads
        ],
    )


def _ledger_has_measured_rounds(ledger_path: Path) -> bool:
    if not ledger_path.is_file():
        return False
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return True
    return bool(ledger.get("rounds"))


def _workspace_has_measured_rounds(workspace: Path) -> bool:
    ledger_paths = [
        workspace / ".ledger.json",
        *workspace.glob("[0-9]*R/agent*/.ledger.json"),
    ]
    return any(_ledger_has_measured_rounds(path) for path in ledger_paths)


def freeze_catalog_evaluation_snapshot(
    workspace: str | Path,
    catalog_name: str,
    definition_name: str,
) -> tuple[CatalogEvaluationSnapshot, Path]:
    """Create once for a new run; reuse the same contract on every resume."""

    root = Path(workspace).resolve()
    path = root / EVALUATION_SNAPSHOT_RELATIVE_PATH
    if path.is_file():
        snapshot = load_catalog_evaluation_snapshot(path)
        expected = (catalog_name, definition_name)
        actual = (snapshot.catalog_name, snapshot.definition_name)
        if actual != expected:
            raise ValueError(
                "workspace evaluation snapshot belongs to a different request: "
                f"snapshot={actual!r}, requested={expected!r}"
            )
        return snapshot, path

    if catalog_name == "kernelswift" and _workspace_has_measured_rounds(root):
        raise RuntimeError(
            "cannot safely resume a measured KernelSwift workspace without its "
            "run-level evaluation snapshot; start a fresh run because the operator "
            "ABI and workload recipe may have changed"
        )

    snapshot = build_catalog_evaluation_snapshot(catalog_name, definition_name)
    return freeze_evaluation_snapshot(root, snapshot)


def freeze_evaluation_snapshot(
    workspace: str | Path,
    requested: CatalogEvaluationSnapshot,
) -> tuple[CatalogEvaluationSnapshot, Path]:
    """Persist an upstream-resolved snapshot, or reuse the workspace copy."""

    root = Path(workspace).resolve()
    path = root / EVALUATION_SNAPSHOT_RELATIVE_PATH
    if path.is_file():
        frozen = load_catalog_evaluation_snapshot(path)
        expected = (requested.catalog_name, requested.definition_name, requested.bundle_id, requested.benchmark_fingerprint)
        actual = (frozen.catalog_name, frozen.definition_name, frozen.bundle_id, frozen.benchmark_fingerprint)
        if actual != expected or (requested.benchmark_fingerprint and frozen != requested):
            raise ValueError(
                "workspace evaluation snapshot belongs to a different request: "
                f"snapshot={actual!r}, requested={expected!r}"
            )
        return frozen, path

    if (
        requested.catalog_name == "kernelswift"
        and _workspace_has_measured_rounds(root)
    ):
        raise RuntimeError(
            "cannot safely resume a measured KernelSwift workspace without its "
            "run-level evaluation snapshot; start a fresh run because the operator "
            "ABI and workload recipe may have changed"
        )

    atomic_write_json(path, requested.model_dump(mode="json"))
    return requested, path


def snapshot_optimization_context(
    snapshot: CatalogEvaluationSnapshot,
) -> tuple[Any, list[dict[str, Any]]]:
    from kernelgen.data.trace import catalog_operator_optimization_context

    definition, correctness, timing = snapshot.native_operator()
    return catalog_operator_optimization_context(
        definition,
        correctness,
        timing,
        group=snapshot.group,
    )


__all__ = [
    "CatalogEvaluationSnapshot",
    "EVALUATION_SNAPSHOT_RELATIVE_PATH",
    "build_catalog_evaluation_snapshot",
    "freeze_catalog_evaluation_snapshot",
    "freeze_evaluation_snapshot",
    "load_catalog_evaluation_snapshot",
    "snapshot_optimization_context",
]
