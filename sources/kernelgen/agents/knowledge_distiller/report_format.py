"""Formatting and validation for per-run distillation reports.

The reports are human/agent-facing artifacts, not canonical KB records.  The
model writes only the analysis body.  Python validates its small Markdown
contract and injects all authoritative run metadata from the ledger.
"""

from __future__ import annotations

import re
from typing import Iterable

from kernelgen.data.optimization_history import OptimizationHistory


EXPERIENCE_HEADINGS = (
    "## Lessons",
    "## Remaining Bottlenecks",
    "## Next Experiments",
)
DETAILED_HEADINGS = (
    "## Strategy Evolution",
    "## Cross-Round Analysis",
    "## Failure Analysis",
    "## Open Questions",
)

_H2_RE = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
_ROUND_REF_RE = re.compile(r"\bR([1-9][0-9]*)\b")


def validate_experience_body(value: str) -> str:
    """Validate the fixed experience fragment emitted by the Distiller."""
    return _validate_body(
        value,
        artifact="candidate_experience",
        expected_headings=EXPERIENCE_HEADINGS,
    )


def validate_detailed_body(value: str) -> str:
    """Validate the fixed detailed-analysis fragment emitted by the Distiller."""
    return _validate_body(
        value,
        artifact="candidate_detailed",
        expected_headings=DETAILED_HEADINGS,
    )


def validate_round_references(
    *,
    experience: str,
    detailed: str,
    valid_rounds: Iterable[int],
) -> None:
    """Require traceable R<n> references and reject references outside ledger."""
    allowed = set(valid_rounds)
    if not allowed:
        if experience.strip() or detailed.strip():
            raise ValueError("distillation reports require measured ledger rounds")
        return
    for name, value in (
        ("candidate_experience", experience),
        ("candidate_detailed", detailed),
    ):
        if not value.strip():
            continue
        referenced = {int(item) for item in _ROUND_REF_RE.findall(value)}
        if not referenced:
            raise ValueError(f"{name} must cite at least one ledger round as R<n>")
        unknown = sorted(referenced - allowed)
        if unknown:
            raise ValueError(
                f"{name} cites rounds absent from ledger: "
                + ", ".join(f"R{item}" for item in unknown)
            )


def render_experience_report(
    body: str,
    history: OptimizationHistory,
) -> str:
    """Render one concise handoff report with Python-owned metadata."""
    return _render_report("Run Experience", body, history)


def render_detailed_report(
    body: str,
    history: OptimizationHistory,
) -> str:
    """Render one cross-round analysis report with Python-owned metadata."""
    return _render_report("Run Analysis", body, history)


def _validate_body(
    value: str,
    *,
    artifact: str,
    expected_headings: tuple[str, ...],
) -> str:
    text = value.strip()
    if not text:
        return ""
    headings = tuple(f"## {match}" for match in _H2_RE.findall(text))
    if headings != expected_headings:
        rendered = " → ".join(expected_headings)
        raise ValueError(
            f"{artifact} must contain exactly these H2 sections in order: {rendered}"
        )
    if re.search(r"^# [^#]", text, flags=re.MULTILINE):
        raise ValueError(f"{artifact} must not include an H1 heading")
    return text


def _render_report(
    title: str,
    body: str,
    history: OptimizationHistory,
) -> str:
    final_status = (
        history.rounds[-1].evaluation.status if history.rounds else "NO_ROUNDS"
    )
    best_round = f"R{history.best_round}" if history.best_round else "none"
    best_geo = (
        f"{history.best_geo_mean:.6g}x" if history.best_round else "none"
    )
    metadata = (
        "## Run Metadata\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        f"| Ledger schema | {_cell(history.schema_version)} |\n"
        f"| Definition | {_cell(history.definition_name)} |\n"
        f"| Hardware | {_cell(history.target_hardware or 'unknown')} |\n"
        f"| Language | {_cell(history.implementation_language or 'unknown')} |\n"
        f"| Measured rounds | {len(history.rounds)} |\n"
        f"| Best round | {best_round} |\n"
        f"| Best geo mean | {best_geo} |\n"
        f"| Final status | {_cell(final_status)} |"
    )
    return f"# {title}\n\n{metadata}\n\n{body.strip()}\n"


def _cell(value: object) -> str:
    """Keep ledger-owned values on one safe Markdown table line."""
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")

