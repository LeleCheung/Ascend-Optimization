#!/usr/bin/env python3
"""Create deterministic coverage samples for a native v6.2 catalog.

The active ``correctness.jsonl`` and ``timing.jsonl`` files are capped while
their complete sources are retained beside them as ``*_full.jsonl``.  Sampling
first covers every individual input value it can, then greedily maximizes
pairwise input-value coverage.  Stable hashes break ties, and selected rows are
written in their original order.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ALGORITHM = "input-value-pairwise-greedy-v1"
PHASES = ("correctness", "timing")


@dataclass(frozen=True)
class WorkloadRow:
    line: str
    value: dict[str, Any]
    stable_key: str


@dataclass(frozen=True)
class SampleResult:
    rows: tuple[WorkloadRow, ...]
    unary_features: int
    covered_unary_features: int
    pair_features: int
    covered_pair_features: int


@dataclass(frozen=True)
class PhasePlan:
    operator_root: Path
    phase: str
    source_is_full: bool
    full_count: int
    sample: SampleResult


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _feature_sets(
    workload: dict[str, Any],
) -> tuple[frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]]:
    inputs = workload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("workload inputs must be an object")

    unary: set[tuple[str, ...]] = set()
    primary: list[tuple[str, str]] = []
    for name in sorted(inputs):
        value = inputs[name]
        encoded = _canonical(value)
        primary.append((name, encoded))
        unary.add(("input", name, encoded))
        if isinstance(value, dict):
            for attribute in sorted(value):
                attribute_value = value[attribute]
                unary.add(
                    ("attribute", name, attribute, _canonical(attribute_value))
                )
                if attribute == "shape" and isinstance(attribute_value, list):
                    unary.add(("shape-rank", name, str(len(attribute_value))))
                    for index, dimension in enumerate(attribute_value):
                        unary.add(
                            ("shape-dimension", name, str(index), _canonical(dimension))
                        )

    tolerance = workload.get("tolerance")
    if tolerance is not None:
        unary.add(("tolerance", _canonical(tolerance)))

    pairwise: set[tuple[str, ...]] = set()
    for left in range(len(primary)):
        left_name, left_value = primary[left]
        for right in range(left + 1, len(primary)):
            right_name, right_value = primary[right]
            pairwise.add(
                ("pair", left_name, left_value, right_name, right_value)
            )
    return frozenset(unary), frozenset(pairwise)


def _read_rows(path: Path) -> tuple[WorkloadRow, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing workload file: {path}") from exc
    rows: list[WorkloadRow] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: workload must be an object")
        if not isinstance(value.get("name"), str) or not value["name"]:
            raise ValueError(f"{path}:{line_number}: workload has no valid name")
        _feature_sets(value)
        rows.append(
            WorkloadRow(
                line=line,
                value=value,
                stable_key=hashlib.sha256(line.encode("utf-8")).hexdigest(),
            )
        )
    if not rows:
        raise ValueError(f"empty workload file: {path}")
    return tuple(rows)


def _intern_features(
    features: Iterable[frozenset[tuple[str, ...]]],
) -> tuple[list[frozenset[int]], int]:
    identifiers: dict[tuple[str, ...], int] = {}
    encoded: list[frozenset[int]] = []
    for feature_set in features:
        values: set[int] = set()
        for feature in feature_set:
            identifier = identifiers.setdefault(feature, len(identifiers))
            values.add(identifier)
        encoded.append(frozenset(values))
    return encoded, len(identifiers)


def _greedy_cover(
    feature_sets: list[frozenset[int]],
    feature_count: int,
    stable_keys: list[str],
    selected: set[int],
    limit: int,
) -> set[int]:
    uncovered = set(range(feature_count))
    for index in selected:
        uncovered.difference_update(feature_sets[index])
    heap: list[tuple[int, str, int]] = []
    for index, features in enumerate(feature_sets):
        if index not in selected:
            gain = sum(feature in uncovered for feature in features)
            heapq.heappush(heap, (-gain, stable_keys[index], index))

    while heap and uncovered and len(selected) < limit:
        negated_bound, stable_key, index = heapq.heappop(heap)
        gain = sum(feature in uncovered for feature in feature_sets[index])
        if gain <= 0:
            if not heap or -heap[0][0] <= 0:
                break
            continue
        if heap:
            next_bound = -heap[0][0]
            next_key = heap[0][1]
            if gain < next_bound or (gain == next_bound and stable_key > next_key):
                heapq.heappush(heap, (-gain, stable_key, index))
                continue
        selected.add(index)
        uncovered.difference_update(feature_sets[index])
    return selected


def sample_rows(rows: tuple[WorkloadRow, ...], limit: int) -> SampleResult:
    """Return at most ``limit`` rows with deterministic feature coverage."""

    if limit <= 0:
        raise ValueError("workload limit must be positive")
    raw_features = [_feature_sets(row.value) for row in rows]
    unary, unary_count = _intern_features(item[0] for item in raw_features)
    pairwise, pair_count = _intern_features(item[1] for item in raw_features)
    stable_keys = [row.stable_key for row in rows]
    selected: set[int] = set()
    target = min(limit, len(rows))

    _greedy_cover(unary, unary_count, stable_keys, selected, target)
    if len(selected) < target:
        _greedy_cover(pairwise, pair_count, stable_keys, selected, target)
    if len(selected) < target:
        remaining = sorted(
            (stable_keys[index], index)
            for index in range(len(rows))
            if index not in selected
        )
        selected.update(index for _, index in remaining[: target - len(selected)])

    selected_rows = tuple(rows[index] for index in sorted(selected))
    covered_unary = set().union(*(unary[index] for index in selected))
    covered_pairwise = set().union(*(pairwise[index] for index in selected))
    return SampleResult(
        rows=selected_rows,
        unary_features=unary_count,
        covered_unary_features=len(covered_unary),
        pair_features=pair_count,
        covered_pair_features=len(covered_pairwise),
    )


def _write_lines(path: Path, rows: Iterable[WorkloadRow]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row.line)
                handle.write("\n")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sample_catalog(root: Path, *, limit: int = 200) -> dict[str, Any]:
    """Sample every operator phase and update the catalog manifest."""

    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("api_version") != "v6.2" or manifest.get("evaluator") != "native":
        raise ValueError("sampling requires a native v6.2 catalog")
    definitions = sorted((root / "ops").rglob("definition.json"))
    if not definitions:
        raise ValueError("catalog contains no operators")

    plans: list[PhasePlan] = []
    operator_stats: dict[str, dict[str, int | str]] = {}
    for definition_path in definitions:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        name = definition.get("name")
        if not isinstance(name, str) or definition_path.parent.name != name:
            raise ValueError(f"invalid operator definition path: {definition_path}")
        relative = definition_path.parent.relative_to(root / "ops")
        group = relative.parts[0] if len(relative.parts) == 2 else ""
        stats: dict[str, int | str] = {"name": name, "group": group}
        for phase in PHASES:
            active = definition_path.parent / f"{phase}.jsonl"
            full = definition_path.parent / f"{phase}_full.jsonl"
            if full.exists() and not full.is_file():
                raise ValueError(f"workload archive is not a file: {full}")
            source_is_full = full.is_file()
            source = full if source_is_full else active
            rows = _read_rows(source)
            sampled = sample_rows(rows, limit)
            plans.append(
                PhasePlan(
                    definition_path.parent,
                    phase,
                    source_is_full,
                    len(rows),
                    sampled,
                )
            )
            stats[f"num_{phase}_workloads"] = len(sampled.rows)
            stats[f"num_{phase}_workloads_full"] = len(rows)
        operator_stats[name] = stats

    sampled_totals = {
        "operators": len(operator_stats),
        "correctness_workloads": sum(
            int(item["num_correctness_workloads"])
            for item in operator_stats.values()
        ),
        "timing_workloads": sum(
            int(item["num_timing_workloads"])
            for item in operator_stats.values()
        ),
    }
    full_totals = {
        "operators": len(operator_stats),
        "correctness_workloads": sum(
            int(item["num_correctness_workloads_full"])
            for item in operator_stats.values()
        ),
        "timing_workloads": sum(
            int(item["num_timing_workloads_full"])
            for item in operator_stats.values()
        ),
    }
    existing_entries = {
        item.get("name"): item
        for item in manifest.get("operators", [])
        if isinstance(item, dict)
    }
    operators: list[dict[str, Any]] = []
    for name, stats in operator_stats.items():
        entry = dict(existing_entries.get(name, {}))
        entry.update(stats)
        operators.append(entry)
    manifest["counts"] = sampled_totals
    manifest["full_counts"] = full_totals
    manifest["sampling"] = {
        "algorithm": ALGORITHM,
        "max_workloads_per_phase": limit,
        "source_files": ["correctness_full.jsonl", "timing_full.jsonl"],
    }
    manifest["operators"] = operators

    staged: list[tuple[Path, Path]] = []
    try:
        for plan in plans:
            active = plan.operator_root / f"{plan.phase}.jsonl"
            staged.append((active, _write_lines(active, plan.sample.rows)))
        staged_manifest = _write_json(manifest_path, manifest)
        for plan, (active, temporary) in zip(plans, staged, strict=True):
            full = plan.operator_root / f"{plan.phase}_full.jsonl"
            if not plan.source_is_full:
                active.replace(full)
            temporary.replace(active)
        staged_manifest.replace(manifest_path)
    except BaseException:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cap active native v6.2 workloads while retaining the complete "
            "JSONL files as *_full.jsonl."
        )
    )
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    manifest = sample_catalog(args.catalog, limit=args.limit)
    print(
        json.dumps(
            {
                "algorithm": manifest["sampling"]["algorithm"],
                "limit": manifest["sampling"]["max_workloads_per_phase"],
                "counts": manifest["counts"],
                "full_counts": manifest["full_counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
