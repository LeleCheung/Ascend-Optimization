"""Validation for runtime and reviewed static knowledge candidates."""

from __future__ import annotations

import re
from typing import Iterable

from kernelgen.knowledge.models import (
    CandidateConcept,
    CandidateDraft,
    RuntimeCandidate,
    SourcePackage,
)
from kernelgen.knowledge.vocabulary import Vocabulary


_STANDARD_CANDIDATE_HEADINGS = (
    "## Claim",
    "## Evidence",
    "## Applicability",
    "## Action",
    "## Limits",
)
_DIAGNOSTIC_CANDIDATE_HEADINGS = (
    "## Symptom",
    "## Likely Causes",
    "## Candidate Techniques",
    "## Diagnosis Checklist",
    "## Caveats",
)
_H2_RE = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
_ROUND_RE = re.compile(r"\bR([1-9][0-9]*)\b")
_ROUND_RANGE_RE = re.compile(
    r"\bR([1-9][0-9]*)\s*(?:-|–|—|\.\.)\s*R?([1-9][0-9]*)\b"
)


def validate_runtime_candidate(
    candidate: CandidateConcept | CandidateDraft | RuntimeCandidate,
) -> None:
    """Validate the small, evidence-oriented contract for runtime knowledge."""
    _validate_candidate_contract(candidate)

    intended = {item.round_num for item in candidate.observation_intents}
    referenced = candidate_round_references(candidate.body)
    if intended and referenced != intended:
        missing = sorted(intended - referenced)
        unknown = sorted(referenced - intended)
        details = []
        if missing:
            details.append(
                "missing intent rounds " + ", ".join(f"R{item}" for item in missing)
            )
        if unknown:
            details.append(
                "rounds without intents " + ", ".join(f"R{item}" for item in unknown)
            )
        raise ValueError("candidate body round references do not match: " + "; ".join(details))
    if not intended and referenced:
        raise ValueError("source-only candidate body must not cite ledger rounds")


def validate_static_candidate(
    candidate: CandidateConcept,
    source_packages: Iterable[SourcePackage],
) -> None:
    """Validate a reviewed source-backed Candidate before publication."""
    if candidate.created_by.endswith(":direct"):
        _validate_direct_candidate_contract(candidate)
    else:
        _validate_candidate_contract(candidate)
    target = candidate.scope.target
    identity_fields = {
        "backend": target.backend,
        "architecture": target.architecture,
        "devices": target.devices,
        "capabilities": target.capabilities,
    }
    if target.level == "portable":
        present = sorted(name for name, value in identity_fields.items() if value)
        if present:
            raise ValueError(
                "portable target scope cannot declare hardware identity: "
                + ", ".join(present)
            )
    elif target.level == "backend":
        if not target.backend:
            raise ValueError("backend target scope requires backend")
        if target.architecture or target.devices:
            raise ValueError(
                "backend target scope cannot declare architecture or devices"
            )
    elif target.level == "architecture":
        if not target.backend or not target.architecture:
            raise ValueError(
                "architecture target scope requires backend and architecture"
            )
        if target.devices:
            raise ValueError(
                "architecture target scope cannot declare devices"
            )
    elif target.level == "device":
        if not target.backend or not target.devices:
            raise ValueError(
                "device target scope requires backend and at least one device"
            )
    elif target.level == "exact":
        if (
            not target.backend
            or not target.architecture
            or len(target.devices) != 1
        ):
            raise ValueError(
                "exact target scope requires backend, architecture, "
                "and exactly one device"
            )

    packages = {package.id: package for package in source_packages}
    for source in candidate.source_refs:
        package = packages.get(source.resource)
        if package is None or not package.allowed_backends:
            continue
        if (
            target.level == "portable"
            or target.backend not in package.allowed_backends
        ):
            raise ValueError(
                f"source {package.id} only permits backends "
                f"{package.allowed_backends}, but Candidate scope is "
                f"level={target.level}, backend={target.backend!r}"
            )


def _validate_direct_candidate_contract(candidate: CandidateConcept) -> None:
    """Validate the source-preserving wrapper used by reviewed direct entries."""
    body = candidate.body.strip()
    if (
        not body.startswith("# Claim\n")
        or "\n# Source material\n" not in body
        or "\n# Applicability\n" not in body
    ):
        raise ValueError(
            "direct candidate body requires Claim, Source material, "
            "and Applicability sections"
        )
    _validate_candidate_routes_and_sources(
        candidate,
        validate_kind_body=False,
    )


def _validate_candidate_contract(
    candidate: CandidateConcept | CandidateDraft | RuntimeCandidate,
) -> None:
    required_headings = (
        _DIAGNOSTIC_CANDIDATE_HEADINGS
        if candidate.proposed_kind == "diagnostic"
        else _STANDARD_CANDIDATE_HEADINGS
    )
    headings = tuple(
        f"## {item}" for item in _H2_RE.findall(candidate.body.strip())
    )
    if headings != required_headings:
        raise ValueError(
            "candidate body must contain exactly these H2 sections in order: "
            + " → ".join(required_headings)
        )
    if re.search(r"^# [^#]", candidate.body, flags=re.MULTILINE):
        raise ValueError("candidate body must not include an H1 heading")
    _validate_candidate_routes_and_sources(candidate)


def _validate_candidate_routes_and_sources(
    candidate: CandidateConcept | CandidateDraft | RuntimeCandidate,
    *,
    validate_kind_body: bool = True,
) -> None:
    if not candidate.retrieval.phases:
        raise ValueError("candidate requires at least one retrieval phase")
    if not candidate.retrieval.tasks:
        raise ValueError("candidate requires at least one retrieval task")
    if candidate.proposed_kind == "diagnostic" and not candidate.retrieval.symptoms:
        raise ValueError("Diagnostic candidate requires at least one symptom")
    if validate_kind_body and candidate.proposed_kind == "method":
        action = _h2_section(candidate.body, "Action")
        limits = _h2_section(candidate.body, "Limits")
        if not re.search(
            r"^### Expected Metric Change\s*$",
            action,
            flags=re.MULTILINE,
        ):
            raise ValueError(
                "Method candidate Action requires "
                "### Expected Metric Change"
            )
        if not re.search(
            r"^### Mechanism Requirements\s*$",
            limits,
            flags=re.MULTILINE,
        ):
            raise ValueError(
                "Method candidate Limits requires "
                "### Mechanism Requirements"
            )
    incomplete_sources = [
        item.resource
        for item in candidate.source_refs
        if not item.revision or not item.locator
    ]
    if incomplete_sources:
        raise ValueError(
            "candidate Source references require revision and locator: "
            + ", ".join(sorted(set(incomplete_sources)))
        )


def _h2_section(body: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$",
        body,
        flags=re.MULTILINE,
    )
    if match is None:
        return ""
    following = _H2_RE.search(body, match.end())
    return body[match.end() : following.start() if following else None]


def normalize_candidate_vocabulary(
    candidate: CandidateConcept | CandidateDraft | RuntimeCandidate,
    vocabulary: Vocabulary,
) -> tuple[
    CandidateConcept | CandidateDraft | RuntimeCandidate,
    list[str],
]:
    """Canonicalize known retrieval values and report unknowns without blocking."""

    techniques, unknown_techniques = vocabulary.normalize_values(
        "techniques",
        candidate.retrieval.techniques,
    )
    symptoms, unknown_symptoms = vocabulary.normalize_values(
        "symptoms",
        candidate.retrieval.symptoms,
    )
    retrieval = candidate.retrieval.model_copy(
        update={
            "techniques": techniques,
            "symptoms": symptoms,
        }
    )
    warnings = [
        f"{candidate.candidate_id}: unknown technique {value!r}"
        for value in unknown_techniques
    ]
    warnings.extend(
        f"{candidate.candidate_id}: unknown symptom {value!r}"
        for value in unknown_symptoms
    )
    return candidate.model_copy(update={"retrieval": retrieval}), warnings


def candidate_round_references(value: str) -> set[int]:
    referenced = {int(item) for item in _ROUND_RE.findall(value)}
    for start_text, end_text in _ROUND_RANGE_RE.findall(value):
        start = int(start_text)
        end = int(end_text)
        if end < start or end - start > 1000:
            raise ValueError(f"invalid candidate round range R{start}-R{end}")
        referenced.update(range(start, end + 1))
    return referenced
