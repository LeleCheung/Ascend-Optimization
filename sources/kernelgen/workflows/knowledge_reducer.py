"""Epoch-level reducer for the legacy Markdown knowledge base.

Agent workspaces only emit ``.new_experience.md`` and ``.new_detailed.md``.
This module is the single writer that turns those candidates into the bounded
run-level KB snapshot consumed by synthesis and the next epoch.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from kernelgen.agents.merge import MergeAgent
from kernelgen.data.ledger import Ledger
from kernelgen.framework.run_control import RunCancelled


EXPERIENCE_LIMIT = 12_000
DETAILED_LIMIT = 30_000
_MERGE_INPUT_LIMITS = {
    "experience": 36_000,
    "detailed": 60_000,
}
_EMPTY_EXPERIENCE_MARKERS = (
    "no experience yet for this kernel",
    "no prior experience",
)


@dataclass(frozen=True)
class KnowledgeCandidate:
    """One agent's proposed KB update plus Python-owned evaluation metadata."""

    agent_id: str
    status: str
    best_geo: float
    best_round: int
    solution_sha: str
    ledger_ref: str
    experience: str = ""
    detailed: str = ""

    def text_for(self, label: str) -> str:
        return self.experience if label == "experience" else self.detailed


def collect_knowledge_candidates(
    results: Iterable[Any],
    workspace: Any,
    *,
    definition_name: str,
    target_hardware: str,
) -> List[KnowledgeCandidate]:
    """Load candidate documents and their authoritative ledger metadata."""

    candidates = []
    for report, workspace_name in results:
        try:
            workspace_path = Path(workspace.path_of(workspace_name))
        except (KeyError, TypeError, AttributeError):
            continue

        ledger_path = workspace_path / ".ledger.json"
        if not ledger_path.is_file():
            continue
        try:
            history = Ledger(workspace_path).history
        except (OSError, ValueError, TypeError):
            continue
        if history.definition_name and history.definition_name != definition_name:
            continue
        if history.target_hardware and history.target_hardware != target_hardware:
            continue

        experience = _read_optional(workspace_path / ".new_experience.md")
        detailed = _read_optional(workspace_path / ".new_detailed.md")
        if not experience and not detailed:
            continue

        best_record = next(
            (
                record
                for record in history.rounds
                if record.round_num == history.best_round
            ),
            None,
        )
        solution_sha = best_record.solution.sha256 if best_record else ""
        if not solution_sha and history.best_code:
            solution_sha = hashlib.sha256(
                history.best_code.encode("utf-8")
            ).hexdigest()

        candidates.append(
            KnowledgeCandidate(
                agent_id=str(workspace_name),
                status=str(getattr(report, "status", "FAILED")),
                best_geo=float(history.best_geo_mean or 0.0),
                best_round=int(history.best_round or 0),
                solution_sha=solution_sha,
                ledger_ref=f"{workspace_name}/.ledger.json",
                experience=experience,
                detailed=detailed,
            )
        )

    return sorted(
        candidates,
        key=lambda item: (
            item.status == "PASSED",
            item.best_geo,
            -item.best_round,
            item.agent_id,
        ),
        reverse=True,
    )


def reduce_epoch_knowledge(
    base_kb: Path,
    *,
    definition: Dict[str, Any],
    target_hardware: str,
    candidates: Sequence[KnowledgeCandidate],
    runtime: Any,
) -> List[Path]:
    """Merge one epoch's candidates into the canonical legacy KB entry."""

    if not candidates:
        return []

    definition_name = _safe_component(str(definition.get("name") or ""), "definition")
    op_type = _safe_component(str(definition.get("op_type") or ""), "operator type")
    hardware = _safe_component(target_hardware, "hardware")
    entry_dir = (
        Path(base_kb)
        / "experience"
        / "by_definition"
        / op_type
        / definition_name
        / hardware
    )

    written = []
    for label, filename, output_limit in (
        ("experience", "experience.md", EXPERIENCE_LIMIT),
        ("detailed", "detailed.md", DETAILED_LIMIT),
    ):
        relevant = [
            candidate
            for candidate in candidates
            if candidate.text_for(label).strip()
        ]
        if not relevant:
            continue

        destination = entry_dir / filename
        existing = _read_optional(destination)
        if label == "experience" and _is_empty_placeholder(existing):
            existing = ""

        merged = _merge_document(
            label=label,
            existing=existing,
            candidates=relevant,
            definition=definition,
            target_hardware=target_hardware,
            output_limit=output_limit,
            runtime=runtime,
        )
        if merged and merged.strip() != _read_optional(destination).strip():
            _atomic_write(destination, merged.rstrip() + "\n")
            written.append(destination)
    return written


def _merge_document(
    *,
    label: str,
    existing: str,
    candidates: Sequence[KnowledgeCandidate],
    definition: Dict[str, Any],
    target_hardware: str,
    output_limit: int,
    runtime: Any,
) -> str:
    if not existing.strip() and len(candidates) == 1:
        return _bounded(candidates[0].text_for(label), output_limit)

    documents = _build_documents(
        label,
        existing,
        candidates,
        _MERGE_INPUT_LIMITS[label],
    )
    best = candidates[0]
    instruction = (
        f"Produce the canonical bounded {label}. The candidate ordering and scores "
        "are authoritative: candidate 1 is the best evaluated run. Prefer claims "
        "supported by higher-scoring PASSED candidates; never infer shapes, constants, "
        "or hardware facts that conflict with the canonical definition below. Treat "
        "conflicting candidate metadata as erroneous, not as evidence of a different "
        "configuration unless the ledger metadata explicitly proves that scope. Do not "
        "generalize one failed parameter value into an untested range. "
        "Deduplicate repeated lessons, retain useful scope conditions and contradictions, "
        f"and keep the complete output below {output_limit} characters.\n\n"
        f"Authoritative best: agent={best.agent_id}, status={best.status}, "
        f"geo_mean={best.best_geo:.12g}, round={best.best_round}, "
        f"solution_sha={best.solution_sha or 'unknown'}.\n\n"
        "<canonical_definition>\n"
        f"{_canonical_definition(definition)}\n"
        "</canonical_definition>"
    )

    try:
        merge_out = MergeAgent().run(
            {
                "mode": "n_way_merge",
                "texts": documents,
                "label": label,
                "definition_name": str(definition.get("name") or ""),
                "op_type": str(definition.get("op_type") or ""),
                "target_hardware": target_hardware,
                "merge_instruction": instruction,
            },
            runtime,
        )
        merged = merge_out.merged_text.strip()
        if not merged or len(merged) > output_limit:
            raise ValueError(f"merged {label} exceeds its output bound")
        return merged
    except RunCancelled:
        raise
    except Exception:
        # Existing KB is the last committed authoritative state. Candidate files
        # remain in agent workspaces, so a transient model failure must not replace it.
        if existing.strip():
            return _bounded(existing, output_limit)
        return _bounded(candidates[0].text_for(label), output_limit)


def _build_documents(
    label: str,
    existing: str,
    candidates: Sequence[KnowledgeCandidate],
    total_limit: int,
) -> List[str]:
    documents = []
    remaining = total_limit
    if existing.strip():
        prior_budget = min(len(existing), max(4_000, total_limit // 4))
        prior = _bounded(existing, prior_budget)
        documents.append("<prior_kb>\n" + prior + "\n</prior_kb>")
        remaining -= len(documents[-1])

    per_candidate = max(2_000, remaining // max(1, len(candidates)))
    for index, candidate in enumerate(candidates, start=1):
        body = _bounded(candidate.text_for(label), per_candidate)
        documents.append(
            "<candidate_metadata>\n"
            f"rank: {index}\n"
            f"agent: {candidate.agent_id}\n"
            f"status: {candidate.status}\n"
            f"best_geo_mean: {candidate.best_geo:.12g}\n"
            f"best_round: {candidate.best_round}\n"
            f"solution_sha: {candidate.solution_sha or 'unknown'}\n"
            f"ledger: {candidate.ledger_ref}\n"
            "</candidate_metadata>\n"
            "<candidate_body>\n"
            f"{body}\n"
            "</candidate_body>"
        )
    return documents


def _canonical_definition(definition: Dict[str, Any]) -> str:
    canonical = {
        key: definition.get(key)
        for key in ("name", "op_type", "axes", "inputs", "outputs", "reference")
        if key in definition
    }
    rendered = json.dumps(canonical, indent=2, ensure_ascii=False, default=str)
    return _bounded(rendered, 12_000)


def _safe_component(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or normalized in {".", ".."} or Path(normalized).name != normalized:
        raise ValueError(f"invalid KB {label}: {value!r}")
    return normalized


def _is_empty_placeholder(content: str) -> bool:
    lowered = content.lower()
    return any(marker in lowered for marker in _EMPTY_EXPERIENCE_MARKERS)


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return ""


def _bounded(content: str, limit: int) -> str:
    content = content.strip()
    if len(content) <= limit:
        return content
    marker = "\n\n[truncated by epoch KB reducer]"
    return content[: max(0, limit - len(marker))].rstrip() + marker


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
