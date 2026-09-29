"""One current, evaluated kernel solution for each exact optimization scope."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kernelgen.data._atomic import atomic_write_json, atomic_write_text
from kernelgen.knowledge.layout import CatalogLayout
from kernelgen.knowledge.models import (
    KnowledgeUsageScope,
    OperatorSignature,
    TargetContext,
)


class SolutionManifest(BaseModel):
    """Metadata for the single current best kernel in one registry slot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    solution_ref: str = Field(min_length=1)
    definition_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    evaluation_scope: KnowledgeUsageScope
    run_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    round_num: int = Field(gt=0)
    archive_ref: str = Field(min_length=1)
    geo_mean: float = Field(gt=0)
    start_mode: Literal["fresh", "fork", "resume"] = "fresh"
    parent_solution_ref: str = ""
    updated_at: datetime


class SolutionSeed(BaseModel):
    """A resolved exact-scope baseline for a new forked run."""

    manifest: SolutionManifest
    code: str = Field(min_length=1)


class SolutionRegistry:
    """Filesystem registry with one replaceable best solution per scope."""

    def __init__(self, catalog_root: Path):
        self.catalog_root = Path(catalog_root)
        self.root = CatalogLayout(self.catalog_root).solutions

    def resolve(
        self,
        signature: OperatorSignature,
        target: TargetContext,
        *,
        benchmark_id: str,
    ) -> SolutionSeed | None:
        manifest_path, code_path = self._paths(signature, target, benchmark_id)
        if not manifest_path.is_file() or not code_path.is_file():
            return None
        manifest = SolutionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        self._validate_match(manifest, signature, target, benchmark_id)
        code = code_path.read_text(encoding="utf-8")
        return SolutionSeed(manifest=manifest, code=code)

    def promote(
        self,
        *,
        signature: OperatorSignature,
        target: TargetContext,
        benchmark_id: str,
        run_id: str,
        workspace_id: str,
        round_num: int,
        archive_ref: str,
        code: str,
        geo_mean: float,
        start_mode: str,
        parent_solution_ref: str = "",
    ) -> SolutionManifest | None:
        """Replace a slot only when a passed result is strictly better."""

        if not code or geo_mean <= 0:
            return None
        manifest_path, code_path = self._paths(signature, target, benchmark_id)
        existing = self._read_manifest(manifest_path)
        if existing is not None:
            self._validate_match(existing, signature, target, benchmark_id)
            if existing.geo_mean >= geo_mean:
                return None
        from kernelgen.knowledge.catalog import (
            assert_catalog_paths_clean,
            catalog_changed_paths,
        )

        assert_catalog_paths_clean(
            self.catalog_root,
            (manifest_path, code_path),
            changed_paths=catalog_changed_paths(self.catalog_root),
        )

        scope = KnowledgeUsageScope.from_context(signature, target)
        manifest = SolutionManifest(
            solution_ref=self._solution_ref(signature, target, benchmark_id),
            definition_id=signature.definition_id,
            benchmark_id=benchmark_id,
            evaluation_scope=scope,
            run_id=run_id,
            workspace_id=workspace_id,
            round_num=round_num,
            archive_ref=archive_ref,
            geo_mean=geo_mean,
            start_mode=start_mode,
            parent_solution_ref=parent_solution_ref,
            updated_at=datetime.now(timezone.utc),
        )
        atomic_write_text(code_path, code)
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
        return manifest

    def _paths(
        self,
        signature: OperatorSignature,
        target: TargetContext,
        benchmark_id: str,
    ) -> tuple[Path, Path]:
        slot = (
            self.root
            / _target_key(target)
            / _safe(signature.definition_id)
            / _safe(benchmark_id)
        )
        return slot / "manifest.json", slot / "best_kernel.py"

    def _solution_ref(
        self,
        signature: OperatorSignature,
        target: TargetContext,
        benchmark_id: str,
    ) -> str:
        return "solution://" + "/".join(
            (
                _target_key(target),
                _safe(signature.definition_id),
                _safe(benchmark_id),
            )
        )

    def _read_manifest(self, path: Path) -> SolutionManifest | None:
        if not path.is_file():
            return None
        return SolutionManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    @staticmethod
    def _validate_match(
        manifest: SolutionManifest,
        signature: OperatorSignature,
        target: TargetContext,
        benchmark_id: str,
    ) -> None:
        expected = KnowledgeUsageScope.from_context(signature, target)
        actual = manifest.evaluation_scope
        if (
            manifest.definition_id != signature.definition_id
            or manifest.benchmark_id != benchmark_id
            or actual.definition_id != expected.definition_id
            or actual.target_backend != expected.target_backend
            or actual.target_architecture != expected.target_architecture
            or actual.target_device != expected.target_device
            or actual.software.get("language", "")
            != expected.software.get("language", "")
        ):
            raise ValueError(
                f"solution manifest scope does not match: {manifest.solution_ref}"
            )


def _target_key(target: TargetContext) -> str:
    language = target.software.language or "unknown"
    architecture = target.architecture or "unknown"
    return "--".join(
        (
            _safe(target.backend),
            _safe(architecture),
            _safe(target.device),
            _safe(language),
        )
    )


def _safe(value: str) -> str:
    normalized = "".join(
        character.lower()
        if character.isalnum() or character in "._-"
        else "-"
        for character in value
    )
    return normalized.strip("-") or "unknown"
