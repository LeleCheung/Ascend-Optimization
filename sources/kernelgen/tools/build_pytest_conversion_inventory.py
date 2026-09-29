#!/usr/bin/env python3
"""Build the serial FlagGems pytest-conversion inventory.

The source CSV uses a mixture of public catalog ids and ATen overload spellings.
This tool preserves the CSV spelling for reporting while resolving a canonical
FlagGems catalog id for ``gems_op`` and pytest discovery.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from kernelgen.agents.extractor.flaggems.source_inventory import (
    collect_source_inventory,
)


CHIPS = (
    "kunlunxin",
    "haiguang",
    "moer",
    "muxi",
    "huawei",
    "tianshu",
    "pingtouge",
    "suiyuan",
    "nvidia",
)

# Source-data spellings that cannot be resolved uniquely from operators.yaml.
SOURCE_ALIASES = {
    "__ilshift__": "ilshift",
    "__rshift__": "rshift",
    "arcsin.out": "arcsin_out",
    "leaky_relu_.out": "leaky_relu_out",
    "lift.out": "lift_out",
}

PYTEST_MARK_ALIASES = {
    "_flash_attention_forward": "underscore_flash_attention_forward",
}


def _normalized_name(value: str) -> str:
    """Normalize overload punctuation while preserving meaningful suffix `_`."""

    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).lstrip("_")


def _unique_catalog_id(entries: list[dict[str, Any]]) -> str | None:
    ids = {str(entry.get("id")) for entry in entries if entry.get("id")}
    if len(ids) == 1:
        return ids.pop()
    return None


def resolve_catalog_id(source_operator: str, entries: list[dict[str, Any]]) -> str:
    """Resolve one CSV spelling to the direct FlagGems public callable id."""

    if source_operator in SOURCE_ALIASES:
        return SOURCE_ALIASES[source_operator]

    lowered = source_operator.lower()
    normalized = _normalized_name(source_operator)
    ranked = (
        [entry for entry in entries if str(entry.get("id", "")).lower() == lowered],
        [
            entry
            for entry in entries
            if any(str(name).lower() == lowered for name in entry.get("for", []))
        ],
        [
            entry
            for entry in entries
            if _normalized_name(str(entry.get("id", ""))) == normalized
        ],
        [
            entry
            for entry in entries
            if any(
                _normalized_name(str(name)) == normalized
                for name in entry.get("for", [])
            )
        ],
    )
    for candidates in ranked:
        resolved = _unique_catalog_id(candidates)
        if resolved is not None:
            return resolved
        if candidates:
            ids = sorted({str(entry.get("id")) for entry in candidates})
            raise ValueError(
                f"ambiguous FlagGems catalog id for {source_operator!r}: {ids}"
            )
    raise ValueError(f"no FlagGems catalog id for {source_operator!r}")


def _load_csv_order(csv_path: Path) -> list[str]:
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        next(rows)
        return [
            row[0].strip()
            for row in rows
            if len(row) >= 19 and row[0].strip() and row[18].strip().isdigit()
        ]


def _load_chip_failures(todo_root: Path) -> dict[str, set[str]]:
    failures: dict[str, set[str]] = {}
    for chip in CHIPS:
        path = todo_root / chip / "failed_ops.txt"
        failures[chip] = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return failures


def _load_ordered_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _relative_paths(paths: tuple[Path, ...], repo: Path) -> list[str]:
    return [path.resolve().relative_to(repo.resolve()).as_posix() for path in paths]


@lru_cache(maxsize=None)
def _pytest_marks(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
        and node.value.attr == "mark"
    }


def _exact_suite_files(
    paths: tuple[Path, ...], operator: str, suite_name: str
) -> tuple[Path, ...]:
    pytest_mark = PYTEST_MARK_ALIASES.get(operator, operator)
    selected = tuple(path for path in paths if pytest_mark in _pytest_marks(path))
    if not selected:
        raise ValueError(
            f"no exact {suite_name} pytest marker {pytest_mark!r} for {operator!r}"
        )
    return selected


def build_inventory(todo_root: Path, flaggems_repo: Path) -> dict[str, Any]:
    """Return a deterministic manifest for all operators failed on any chip."""

    csv_path = todo_root / "20260826.csv"
    source_order = _load_csv_order(csv_path)
    # NVIDIA has no dedicated result column in the CSV.  Its later 72-op
    # backlog is an explicit supplemental source and keeps its reviewed order.
    for operator in _load_ordered_lines(todo_root / "nvidia" / "failed_ops.txt"):
        if operator not in source_order:
            source_order.append(operator)
    chip_failures = _load_chip_failures(todo_root)
    failed_union = set().union(*chip_failures.values())

    catalog = yaml.safe_load(
        (flaggems_repo / "conf" / "operators.yaml").read_text(encoding="utf-8")
    )
    entries = catalog.get("ops", catalog)
    if not isinstance(entries, list):
        raise ValueError("FlagGems conf/operators.yaml does not contain an ops list")

    operators: list[dict[str, Any]] = []
    for source_operator in source_order:
        if source_operator not in failed_union:
            continue
        try:
            operator = resolve_catalog_id(source_operator, entries)
        except ValueError:
            # The NVIDIA backlog added candidate-ready pytest before these
            # operators had public FlagGems implementations or operators.yaml
            # entries.  Their exact pytest marker is the stable resolver key.
            if source_operator not in chip_failures["nvidia"]:
                raise
            exact_accuracy = flaggems_repo / "tests" / f"test_{source_operator}.py"
            exact_benchmark = (
                flaggems_repo / "benchmark" / f"test_{source_operator}.py"
            )
            if not exact_accuracy.is_file() or not exact_benchmark.is_file():
                raise
            operator = source_operator
        inventory = collect_source_inventory(str(flaggems_repo), operator)
        accuracy_files = _exact_suite_files(
            inventory.test_files, operator, "accuracy"
        )
        benchmark_files = _exact_suite_files(
            inventory.benchmark_files, operator, "benchmark"
        )
        operators.append(
            {
                "source_operator": source_operator,
                "operator": operator,
                "pytest_mark": PYTEST_MARK_ALIASES.get(operator, operator),
                "chips": [
                    chip
                    for chip in CHIPS
                    if source_operator in chip_failures[chip]
                ],
                "accuracy_files": _relative_paths(accuracy_files, flaggems_repo),
                "benchmark_files": _relative_paths(
                    benchmark_files, flaggems_repo
                ),
            }
        )

    missing = failed_union - {item["source_operator"] for item in operators}
    if missing:
        raise ValueError(f"failed operators missing from valid CSV rows: {sorted(missing)}")

    file_owners: dict[str, list[str]] = defaultdict(list)
    for item in operators:
        for path in [*item["accuracy_files"], *item["benchmark_files"]]:
            file_owners[path].append(item["operator"])
    for item in operators:
        item["shared_files"] = {
            path: file_owners[path]
            for path in [*item["accuracy_files"], *item["benchmark_files"]]
            if len(file_owners[path]) > 1
        }

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=flaggems_repo,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "schema_version": "kernelgen.pytest-conversion/v1",
        "source_csv": csv_path.name,
        "additional_sources": ["nvidia/failed_ops.txt"],
        "flaggems_commit": commit,
        "operator_count": len(operators),
        "chip_entry_count": sum(len(values) for values in chip_failures.values()),
        "operators": operators,
    }


def main() -> None:
    kernelgen_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build all_failed_ops.txt and the FlagGems pytest inventory"
    )
    parser.add_argument(
        "--todo-root",
        type=Path,
        default=kernelgen_root / "kernel_todo_v2",
    )
    parser.add_argument(
        "--flaggems-repo",
        type=Path,
        default=kernelgen_root.parent / "FlagGems-master",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=kernelgen_root
        / "kernel_todo_v2"
        / "pytest_conversion_inventory.json",
    )
    args = parser.parse_args()

    manifest = build_inventory(args.todo_root.resolve(), args.flaggems_repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    all_ops_path = args.todo_root / "all_failed_ops.txt"
    all_ops_path.write_text(
        "".join(f"{item['source_operator']}\n" for item in manifest["operators"]),
        encoding="utf-8",
    )
    print(
        f"wrote {manifest['operator_count']} operators to {args.output} "
        f"and {all_ops_path}"
    )


if __name__ == "__main__":
    main()
