#!/usr/bin/env python3
"""Update a kernel_todo Markdown table from normalized validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_operators(path: Path) -> list[str]:
    operators: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "算子名称" or set(cells[0]) <= {"-", ":"}:
            continue
        operators.append(cells[0])
    if len(operators) != len(set(operators)):
        raise ValueError(f"{path}: duplicate operator names")
    return operators


def render_table(operators: list[str], results: dict[str, dict[str, str]]) -> str:
    missing = sorted(set(operators) - set(results))
    extra = sorted(set(results) - set(operators))
    if missing or extra:
        raise ValueError(f"result mismatch: missing={missing}, extra={extra}")

    lines = [
        "| 算子名称 | 验证结果 | 失败原因 |",
        "| --- | --- | --- |",
    ]
    for operator in operators:
        result = results[operator]
        status = result.get("status", "")
        reason = result.get("reason", "")
        if status not in {"通过", "未通过"}:
            raise ValueError(f"{operator}: invalid status {status!r}")
        if status == "通过":
            reason = "—"
        elif not (
            reason.startswith("[抽取问题]")
            or reason.startswith("[API 不支持]")
        ):
            raise ValueError(f"{operator}: unclassified failure reason {reason!r}")
        if "|" in reason or "\n" in reason:
            raise ValueError(f"{operator}: unsafe Markdown reason")
        lines.append(f"| {operator} | {status} | {reason} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    operators = read_operators(args.markdown)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    rendered = render_table(operators, results)
    temporary = args.markdown.with_suffix(args.markdown.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
