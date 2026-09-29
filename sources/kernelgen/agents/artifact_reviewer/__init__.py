"""Read-only semantic review; orchestration owns artifact identity and acceptance."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from kernelgen.framework.base import BaseAgent
from kernelgen.agents.catalog_blocker import CatalogBlocker


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["catalog", "tests", "code"]
    operator: str
    subject_sha256: str
    evidence_paths: list[str]


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priority: Literal["P0", "P1"]
    category: Literal["conversion", "test_contract", "source_quality", "target_capability"] = Field(
        default="conversion",
        description="Finding scope: conversion fidelity, invalid test contract, source-test quality, or target readiness. Unclassified findings remain conversion issues.",
    )
    evidence: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    requested_change: str = Field(min_length=1)


class ReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    reviewed_files: list[str] = Field(min_length=1)
    findings: list[ReviewFinding]
    blocker: CatalogBlocker | None = None
    missing_dependencies: list[str] = Field(default_factory=list)

    def blocks_catalog(self) -> bool:
        """Block on extraction blockers or P0 conversion defects, not advice."""
        return self.blocker is not None or any(f.priority == "P0" and f.category == "conversion" for f in self.findings)

    def blocks_tests(self) -> bool:
        """Gate invalid test contracts and missing evidence, not coverage advice."""
        return bool(self.blocker or self.missing_dependencies) or any(
            f.priority == "P0" and f.category in {"conversion", "test_contract"}
            for f in self.findings
        )


class ArtifactReviewerAgent(BaseAgent):
    name = "artifact_reviewer"
    InputModel = ReviewInput
    OutputModel = ReviewOutput
