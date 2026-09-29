"""Load a V6 catalog and resolve its evaluator-owned assets."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .protocol.schema import (
    Definition,
    EvaluateRequest,
    EvaluationSettings,
    Implementation,
    Workload,
)
from .protocol.version import KERNELGEN_SUPPORTED_API_VERSIONS


EvaluatorKind = Literal["native", "flaggems"]
FrameworkKind = Literal["flaggems"]


def _flaggems_binding(manifest: dict[str, Any]) -> tuple[str, str, str]:
    if "framework_base_revision" in manifest or (
        "compatible_framework_revisions" in manifest
    ):
        raise ValueError(
            "FlagGems catalogs must use one exact framework_revision; "
            "base revisions and revision allowlists are not supported"
        )
    repository = manifest.get("framework_repository")
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("FlagGems catalogs must declare framework_repository")
    branch = manifest.get("framework_branch")
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("FlagGems catalogs must declare framework_branch")
    revision = manifest.get("framework_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "FlagGems catalogs must pin framework_revision to a full "
            "lowercase commit hash"
        )
    return repository, branch, revision


@dataclass(frozen=True)
class OperatorData:
    record_id: str
    relative: str
    definition: Definition
    evaluator: EvaluatorKind
    correctness_workloads: list[Workload]
    timing_workloads: list[Workload]
    oracle_path: Path | None = None
    assets_digest: str | None = None


class Catalog:
    """A single-evaluator catalog discovered from its definitions directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        api_version = self.manifest.get("api_version")
        if api_version not in KERNELGEN_SUPPORTED_API_VERSIONS:
            raise ValueError("unsupported catalog API version")
        self.api_version = api_version
        evaluator = self.manifest.get("evaluator")
        if evaluator not in {"native", "flaggems"}:
            raise ValueError("catalog manifest evaluator must be native or flaggems")
        self.evaluator: EvaluatorKind = evaluator
        self.framework: FrameworkKind | None = None
        self.framework_root: Path | None = None
        self.framework_repository: str | None = None
        self.framework_branch: str | None = None
        self.framework_revision: str | None = None
        if self.evaluator == "flaggems":
            if self.manifest.get("benchmark_level") != "core":
                raise ValueError("FlagGems catalogs must use benchmark_level=core")
            # Adapter assets come from the selected runtime checkout, not a
            # catalog pin. Legacy framework fields do not bind this evaluator.

        layout = self.manifest.get("layout")
        if api_version == "v6.2":
            if self.evaluator != "native" or layout != "per-operator":
                raise ValueError(
                    "v6.2 catalogs require evaluator=native and layout=per-operator"
                )
            framework = self.manifest.get("framework")
            if framework not in {None, "flaggems"}:
                raise ValueError(
                    "v6.2 native catalog framework must be flaggems when set"
                )
            if framework == "flaggems":
                self.framework = "flaggems"
                (
                    self.framework_repository,
                    self.framework_branch,
                    self.framework_revision,
                ) = _flaggems_binding(self.manifest)
            definitions_root = self.root / "ops"
            definition_paths = sorted(definitions_root.rglob("definition.json"))
        else:
            if layout not in {None, "flat"}:
                raise ValueError("v6.0 catalogs use the flat definitions layout")
            definitions_root = self.root / "definitions"
            definition_paths = sorted(definitions_root.rglob("*.json"))
        if not definitions_root.is_dir():
            raise ValueError("catalog definitions directory is missing")
        entries: dict[str, tuple[str, Path, Definition]] = {}
        for path in definition_paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"definition must be a JSON object: {path}")
            if api_version == "v6.2":
                runtime_fields = {"reference", "reference_device"} & set(raw)
                if runtime_fields:
                    raise ValueError(
                        "v6.2 definition.json must contain only the public ABI; "
                        f"remove runtime fields: {sorted(runtime_fields)}"
                    )
            definition = Definition.model_validate(raw)
            if definition.api_version != api_version:
                raise ValueError("definition API version must match catalog manifest")
            if api_version == "v6.2":
                relative = path.parent.relative_to(definitions_root).as_posix()
                relative_parts = Path(relative).parts
                if (
                    len(relative_parts) not in {1, 2}
                    or path.parent.name != definition.name
                ):
                    raise ValueError(
                        "v6.2 operator directory must be "
                        "ops/<Definition.name> or "
                        "ops/<group>/<Definition.name>"
                    )
            else:
                relative = path.relative_to(definitions_root).with_suffix("").as_posix()
            if definition.name in entries:
                raise ValueError(
                    f"catalog contains duplicate Definition.name: {definition.name}"
                )
            entries[definition.name] = (relative, path, definition)
        if not entries:
            raise ValueError("catalog contains no definitions")
        self._entries = entries

    @property
    def operator_names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def _paired_path(self, directory: str, relative: str, suffix: str) -> Path:
        root = (self.root / directory).resolve()
        path = (root / f"{relative}{suffix}").resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                f"missing native asset: {directory}/{relative}{suffix}"
            )
        return path

    @staticmethod
    def _load_workloads(path: Path) -> list[Workload]:
        return [
            Workload.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _oracle_reference_device(path: Path) -> str:
        try:
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"invalid oracle.py syntax: {exc}") from exc
        values: list[ast.expr | None] = []
        for node in module.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "REFERENCE_DEVICE"
                for target in node.targets
            ):
                values.append(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "REFERENCE_DEVICE"
            ):
                values.append(node.value)
        if len(values) != 1:
            raise ValueError("oracle.py must declare REFERENCE_DEVICE exactly once")
        try:
            value = ast.literal_eval(values[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("oracle.py REFERENCE_DEVICE must be a string literal") from exc
        if value not in {"target", "cpu"}:
            raise ValueError("oracle.py REFERENCE_DEVICE must be 'target' or 'cpu'")
        return value

    def _load_per_operator(
        self,
        name: str,
        relative: str,
        path: Path,
        definition: Definition,
    ) -> OperatorData:
        operator_root = path.parent
        forbidden = [
            filename
            for filename in (
                "correctness_reference.py",
                "timing_reference.py",
                "torch_reference.py",
            )
            if (operator_root / filename).exists()
        ]
        if forbidden:
            raise ValueError(
                "v6.2 keeps run, correctness_run, timing_run and torch_run in "
                "oracle.py; "
                f"separate reference files are unsupported: {forbidden}"
            )
        oracle_path = operator_root / "oracle.py"
        if not oracle_path.is_file():
            raise ValueError(f"missing native asset: ops/{relative}/oracle.py")
        assets_digest = self._validate_operator_assets(operator_root, relative)
        source = oracle_path.read_text(encoding="utf-8")
        reference_device = self._oracle_reference_device(oracle_path)
        public_payload = definition.model_dump(mode="python", exclude_unset=True)
        public_payload["api_version"] = definition.api_version
        definition = Definition.model_validate(
            {
                **public_payload,
                "reference": source,
                "reference_device": reference_device,
            }
        )
        correctness_path = operator_root / "correctness.jsonl"
        timing_path = operator_root / "timing.jsonl"
        correctness = (
            self._load_workloads(correctness_path) if correctness_path.is_file() else []
        )
        timing = self._load_workloads(timing_path) if timing_path.is_file() else []
        if not correctness and not timing:
            raise ValueError(f"operator has no workloads: {name}")
        return OperatorData(
            name,
            relative,
            definition,
            "native",
            correctness,
            timing,
            oracle_path,
            assets_digest,
        )

    @staticmethod
    def _validate_operator_assets(operator_root: Path, relative: str) -> str:
        """Validate and fingerprint optional operator-private runtime assets."""

        known = {
            "definition.json",
            "oracle.py",
            "correctness.jsonl",
            "timing.jsonl",
            "correctness_full.jsonl",
            "timing_full.jsonl",
        }
        unexpected = sorted(
            path.name
            for path in operator_root.iterdir()
            if path.name not in known and path.name != "assets"
        )
        if unexpected:
            raise ValueError(
                f"unexpected files in ops/{relative}; put oracle dependencies "
                f"under assets/: {unexpected}"
            )

        for filename in ("correctness_full.jsonl", "timing_full.jsonl"):
            archive = operator_root / filename
            if archive.exists() and not archive.is_file():
                raise ValueError(
                    f"ops/{relative}/{filename} must be a regular file"
                )

        for path in operator_root.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    f"operator assets cannot contain symlinks: "
                    f"ops/{relative}/{path.relative_to(operator_root).as_posix()}"
                )

        assets_root = operator_root / "assets"
        if assets_root.exists() and not assets_root.is_dir():
            raise ValueError(f"ops/{relative}/assets must be a directory")

        digest = hashlib.sha256()
        if assets_root.is_dir():
            for path in sorted(item for item in assets_root.rglob("*") if item.is_file()):
                asset_relative = path.relative_to(assets_root).as_posix()
                digest.update(asset_relative.encode("utf-8"))
                digest.update(b"\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
        return "sha256:" + digest.hexdigest()

    def load(self, name: str) -> OperatorData:
        try:
            relative, definition_path, public_definition = self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown operator: {name}") from exc

        if self.evaluator == "flaggems":
            return OperatorData(name, relative, public_definition, "flaggems", [], [])

        if self.api_version == "v6.2":
            return self._load_per_operator(
                name, relative, definition_path, public_definition
            )

        reference_path = self._paired_path("references", relative, ".py")
        correctness_path = self._paired_path(
            "workloads", relative, ".correctness.jsonl"
        )
        timing_path = self._paired_path("workloads", relative, ".timing.jsonl")
        reference_source = reference_path.read_text(encoding="utf-8")
        reference_device = "target"
        for line in reference_source.splitlines():
            if line.strip().startswith("REFERENCE_DEVICE"):
                _, value = line.split("=", 1)
                reference_device = value.strip().strip("\"'")
                break
        public_payload = public_definition.model_dump(mode="python", exclude_unset=True)
        public_payload["api_version"] = public_definition.api_version
        definition = Definition.model_validate(
            {
                **public_payload,
                "reference": reference_source,
                "reference_device": reference_device,
            }
        )
        correctness = self._load_workloads(correctness_path)
        timing = self._load_workloads(timing_path)
        if not correctness and not timing:
            raise ValueError(f"operator has no workloads: {name}")
        return OperatorData(
            name,
            relative,
            definition,
            "native",
            correctness,
            timing,
        )

    def request(
        self,
        name: str,
        implementation: Implementation,
        settings: EvaluationSettings | None = None,
    ) -> EvaluateRequest:
        operator = self.load(name)
        if operator.evaluator != "native":
            raise ValueError("framework catalogs cannot produce native EvaluateRequest")
        return EvaluateRequest(
            api_version=operator.definition.api_version,
            definition=operator.definition,
            implementation=implementation,
            correctness_workloads=operator.correctness_workloads,
            timing_workloads=operator.timing_workloads,
            settings=settings or EvaluationSettings(),
        )


__all__ = [
    "Catalog",
    "EvaluatorKind",
    "FrameworkKind",
    "OperatorData",
]
