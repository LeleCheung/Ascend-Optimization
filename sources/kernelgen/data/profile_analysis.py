"""Backend-neutral, evidence-carrying profile analysis contract."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProfileAnalysisStatus = Literal["completed", "inconclusive", "unsupported", "failed"]
DominantBound = Literal["compute", "memory", "latency", "launch", "mixed", "unknown"]
Confidence = Literal["high", "medium", "low"]


class EvidenceLocator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kernel: str = ""
    source_file: str = ""
    source_line: Optional[int] = None
    pc: str = ""


class ProfileEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    value: Any
    unit: str = ""
    artifact_path: str = Field(min_length=1)
    artifact_kind: str = Field(min_length=1)
    locator: EvidenceLocator = Field(default_factory=EvidenceLocator)


class ProfileFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "launch_parallelism",
        "memory_bandwidth",
        "memory_latency",
        "memory_access_efficiency",
        "cache_behavior",
        "resource_pressure",
        "compute_pipeline",
        "synchronization",
        "control_divergence",
        "instruction_efficiency",
        "unknown",
    ]
    label: str = Field(min_length=1)
    backend_detail: str = ""
    workload_uuids: List[str] = Field(min_length=1)
    confidence: Confidence
    evidence: List[ProfileEvidence] = Field(min_length=1)
    inference: str = Field(min_length=1)


class ProfiledWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uuid: str = Field(min_length=1)
    axes: Dict[str, Any] = Field(default_factory=dict)
    eval_speedup: Optional[float] = None
    eval_latency_ms: Optional[float] = None
    eval_reference_latency_ms: Optional[float] = None
    selection_reason: str = Field(min_length=1)
    profile_ids: List[str] = Field(default_factory=list)
    manifest_paths: List[str] = Field(default_factory=list)
    status: Literal["completed", "unsupported", "failed"]

    @model_validator(mode="after")
    def validate_completed_evidence(self) -> "ProfiledWorkload":
        if self.status == "completed" and (not self.profile_ids or not self.manifest_paths):
            raise ValueError("completed profiled workload requires profile_ids and manifest_paths")
        return self


class NextExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_category: str = Field(min_length=1)
    action_description: str = Field(min_length=1)
    expected_impact: str = Field(min_length=1)
    risks_and_rollback: str = Field(min_length=1)
    validation_workloads: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(min_length=1)


class ProfileAnalysis(BaseModel):
    """The only payload allowed to move a round's profile state to terminal."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    round_num: int = Field(gt=0)
    evaluation_fingerprint: str = Field(min_length=1)
    status: ProfileAnalysisStatus
    solution_sha256: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    profiler: str = ""
    capabilities: List[str] = Field(default_factory=list)
    dominant_bound: DominantBound = "unknown"
    profiled_workloads: List[ProfiledWorkload] = Field(default_factory=list)
    findings: List[ProfileFinding] = Field(default_factory=list)
    performance_interpretation: str = ""
    next_experiment: Optional[NextExperiment] = None
    warnings: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    error: str = ""

    @model_validator(mode="after")
    def validate_status_requirements(self) -> "ProfileAnalysis":
        if self.status == "completed":
            if not self.profiled_workloads:
                raise ValueError("completed analysis requires at least one profiled workload")
            if not self.findings:
                raise ValueError("completed analysis requires at least one finding")
            if self.next_experiment is None:
                raise ValueError("completed analysis requires next_experiment")
            if not self.performance_interpretation.strip():
                raise ValueError("completed analysis requires performance_interpretation")
        elif self.status == "inconclusive":
            if not self.profiled_workloads:
                raise ValueError("inconclusive analysis requires collected workload evidence")
            if not self.performance_interpretation.strip():
                raise ValueError("inconclusive analysis must explain why evidence is insufficient")
        elif self.status == "unsupported":
            if not self.capabilities and not self.error.strip():
                raise ValueError("unsupported analysis requires capabilities or an error")
        elif self.status == "failed" and not self.error.strip():
            raise ValueError("failed analysis requires error")
        return self

    def compact_summary(self) -> str:
        if self.status == "completed":
            labels = ", ".join(finding.label for finding in self.findings[:3])
            return f"{self.dominant_bound}: {labels}" if labels else self.dominant_bound
        detail = self.error or self.performance_interpretation
        return f"{self.status}: {detail}"[:500]
