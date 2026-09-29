"""Immutable source packages and reviewed ingestion plans."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from kernelgen.knowledge.models.base import (
    SCHEMA_VERSION,
    ConceptKind,
    StrictModel,
)
from kernelgen.knowledge.models.concept import (
    ConceptRelation,
    RetrievalMetadata,
    Scope,
)
from kernelgen.knowledge.source_rules import source_path_matches


class SourcePackage(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    id: str = Field(pattern=r"^source:[a-z0-9][a-z0-9._-]*$")
    source_type: Literal[
        "documentation",
        "blog",
        "repository",
        "paper",
        "dataset",
        "internal",
    ]
    origin: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    content_root: str = Field(min_length=1)
    license: str = Field(min_length=1)
    checksum: str = Field(default="", pattern=r"^(|sha256:[0-9a-f]{64})$")
    search_mode: Literal["searchable", "static_only"]
    manifest_root: str = ""
    retrieved_at: datetime
    authority: Literal["official", "first_party", "community", "unknown"] = "unknown"
    allowed_hosts: List[str] = Field(default_factory=list)
    usage_policy: Literal[
        "redistributable",
        "restricted",
        "federated_only",
        "unknown",
    ] = "unknown"
    usage_notes: str = ""
    allowed_backends: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_search_mode(cls, value):
        if isinstance(value, dict) and "search_mode" not in value:
            value = dict(value)
            value["search_mode"] = (
                "searchable"
                if value.get("manifest_root")
                else "static_only"
            )
        return value

    @model_validator(mode="after")
    def validate_usage_policy(self) -> "SourcePackage":
        if self.usage_policy in {"restricted", "federated_only"} and not self.usage_notes:
            raise ValueError(f"{self.usage_policy} source requires usage_notes")
        if self.usage_policy == "restricted" and not self.allowed_backends:
            raise ValueError("restricted source requires allowed_backends")
        if self.search_mode == "searchable":
            if not self.manifest_root:
                raise ValueError(
                    "searchable source requires manifest_root"
                )
            if not self.content_root.startswith("kb://"):
                raise ValueError(
                    "searchable source requires local kb:// content_root"
                )
            if not self.manifest_root.startswith("kb://"):
                raise ValueError(
                    "searchable source requires local kb:// manifest_root"
                )
        elif self.manifest_root:
            raise ValueError(
                "static_only source cannot declare manifest_root"
            )
        return self


class GitSourceEntry(StrictModel):
    path: str = Field(min_length=1)
    mode: Literal["100644", "100755", "120000", "160000"]
    object_type: Literal["blob", "commit"]
    object_id: str = Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")
    size: Optional[int] = Field(default=None, ge=0)
    sha256: str = Field(default="", pattern=r"^(|sha256:[0-9a-f]{64})$")

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("Git source path must stay within the package")
        return value

    @model_validator(mode="after")
    def validate_git_object(self) -> "GitSourceEntry":
        if self.object_type == "commit":
            if self.mode != "160000":
                raise ValueError("Git commit entry must use mode 160000")
            if self.size is not None or self.sha256:
                raise ValueError("Git commit entry cannot carry file content")
        else:
            if self.mode == "160000":
                raise ValueError("Git blob entry cannot use mode 160000")
            if self.size is None or not self.sha256:
                raise ValueError("Git blob entry requires size and sha256")
        return self


class GitSourceManifest(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    source_package: str = Field(pattern=r"^source:[a-z0-9][a-z0-9._-]*$")
    revision: str = Field(min_length=1)
    object_format: Literal["sha1", "sha256"]
    entries: List[GitSourceEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entry_order(self) -> "GitSourceManifest":
        paths = [entry.path for entry in self.entries]
        if paths != sorted(paths):
            raise ValueError("Git source manifest entries must be path-sorted")
        if len(paths) != len(set(paths)):
            raise ValueError("Git source manifest paths must be unique")
        return self


IngestionMode = Literal[
    "source_only",
    "indexed",
    "direct",
    "extract",
    "federated",
]


class SourceSelectionRule(StrictModel):
    include: str = Field(min_length=1)
    mode: IngestionMode
    kinds: List[ConceptKind] = Field(default_factory=list)

    @field_validator("include")
    @classmethod
    def validate_relative_pattern(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source selection pattern must stay within the package")
        return value

    @model_validator(mode="after")
    def validate_kind_policy(self) -> "SourceSelectionRule":
        if self.mode in {"direct", "extract"} and not self.kinds:
            raise ValueError(f"{self.mode} selection requires allowed kinds")
        if self.mode in {"source_only", "indexed", "federated"} and self.kinds:
            raise ValueError(f"{self.mode} selection cannot publish kinds")
        if "experience" in self.kinds:
            raise ValueError("static sources cannot publish Experience")
        return self


class StaticKnowledgeEntry(StrictModel):
    source_path: str = Field(min_length=1)
    heading: str = ""
    mode: Literal["direct", "extract"]
    proposed_kind: Literal["reference", "method", "diagnostic"]
    proposed_id: Optional[str] = Field(
        default=None,
        pattern=r"^kg:(reference|method|diagnostic):[a-z0-9][a-z0-9._-]*$",
    )
    claim_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    domains: List[str] = Field(min_length=1)
    scope: Scope
    retrieval: RetrievalMetadata = Field(default_factory=RetrievalMetadata)
    relations: List[ConceptRelation] = Field(default_factory=list)
    body: str = ""

    @field_validator("source_path")
    @classmethod
    def validate_relative_source_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source_path must stay within the package")
        return value

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "StaticKnowledgeEntry":
        if self.mode == "direct" and self.body:
            raise ValueError("direct entry body must come from the source")
        if self.mode == "extract" and not self.body.strip():
            raise ValueError("extract entry requires a reviewed body")
        if self.proposed_id and not self.proposed_id.startswith(
            f"kg:{self.proposed_kind}:"
        ):
            raise ValueError("proposed_id prefix does not match proposed_kind")
        return self


class StaticIngestionPlan(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    source_package: str = Field(
        pattern=r"^source:[a-z0-9][a-z0-9._-]*$"
    )
    rules: List[SourceSelectionRule] = Field(min_length=1)
    entries: List[StaticKnowledgeEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_entries_are_selected(self) -> "StaticIngestionPlan":
        for entry in self.entries:
            matching = [
                rule
                for rule in self.rules
                if source_path_matches(entry.source_path, rule.include)
            ]
            if not matching:
                raise ValueError(
                    f"entry is not covered by a rule: {entry.source_path}"
                )
            rule = matching[-1]
            if rule.mode != entry.mode:
                raise ValueError(
                    f"entry mode does not match rule for {entry.source_path}"
                )
            if entry.proposed_kind not in rule.kinds:
                raise ValueError(
                    f"entry kind is not allowed for {entry.source_path}"
                )
        return self
