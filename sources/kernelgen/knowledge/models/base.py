"""Pydantic contracts for KernelGen Knowledge Base V1.

These models are the machine-readable counterpart of
``docs/design/knowledge/knowledge_base_implementation.md``. They
validate normalized records.  Markdown concepts are parsed by
``kernelgen.knowledge.validation`` before they reach ``Concept``.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

SCHEMA_VERSION = "1.0"
_VERSION_CONSTRAINT = re.compile(r"^[A-Za-z0-9_.+!<>=~,* -]+$")
_TYPE_TO_KIND = {
    "Reference": "reference",
    "Method": "method",
    "Diagnostic": "diagnostic",
    "Experience": "experience",
}

ConceptKind = Literal["reference", "method", "diagnostic", "experience"]
ConceptStatus = Literal["draft", "stable", "deprecated"]
PublishAction = Literal["create", "update"]
EvidenceState = Literal[
    "source_supported",
    "observed",
    "corroborated",
    "contested",
    "falsified",
]
EvidenceStance = Literal["supports", "refutes", "illustrates"]
EvidenceConfidence = Literal["high", "medium", "low"]
TargetLevel = Literal["exact", "device", "architecture", "backend", "portable"]
WorkloadOperator = Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in"]
QueryPhase = Literal[
    "initial",
    "post_error",
    "post_evaluation",
    "post_profile",
    "plateau",
]
QueryTask = Literal[
    "constraint_check",
    "architecture_selection",
    "implementation",
    "diagnosis",
    "next_experiment",
    "portability_analysis",
]
KnowledgeUseRole = Literal[
    "constraint",
    "hypothesis",
    "implementation",
    "diagnostic",
]
KnowledgeUseDisposition = Literal["adopted", "adapted", "rejected"]
KnowledgeUsageMode = Literal["none", "single", "combined"]
KnowledgeApplicationEffect = Literal[
    "correctness_recovered",
    "correctness_regressed",
    "performance_improved",
    "performance_regressed",
    "no_material_change",
    "unclassified",
]
KnowledgeAssessmentStatus = Literal[
    "confirmed",
    "partially_confirmed",
    "not_confirmed",
    "inconclusive",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
