"""Summarize collected BatchSimpleOpt workspaces across accelerator targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from kernelgen.tools.monitor_batch_simple_opt import (
    DefinitionProgress,
    collect_progress,
)


DEFAULT_DEVICE_ORDER = (
    "tianshu",
    "hygon",
    "musa",
    "metax",
    "kunlun",
    "ascend",
    "ppu",
)


@dataclass(frozen=True)
class DeviceReport:
    name: str
    workspace: Path
    definitions: tuple[DefinitionProgress, ...]

    @property
    def passed(self) -> int:
        return sum(item.state == "PASSED" for item in self.definitions)

    @property
    def failed(self) -> int:
        return sum(item.state == "FAILED" for item in self.definitions)

    @property
    def accelerated(self) -> int:
        return sum(
            item.state == "PASSED"
            and item.best_geo_mean is not None
            and item.best_geo_mean > 1.0
            for item in self.definitions
        )


def discover_reports(root: Path, run_name: str) -> list[DeviceReport]:
    reports = []
    candidates = list(DEFAULT_DEVICE_ORDER)
    candidates.extend(
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name not in candidates
    )
    for name in candidates:
        workspace = root / name / run_name / "workspace"
        if workspace.is_dir():
            reports.append(load_report(name, workspace))
    return reports


def load_report(name: str, workspace: Path) -> DeviceReport:
    workspace = workspace.resolve()
    return DeviceReport(
        name=name,
        workspace=workspace,
        definitions=tuple(collect_progress(workspace)),
    )


def definition_names(reports: Iterable[DeviceReport]) -> list[str]:
    names = []
    for report in reports:
        for item in report.definitions:
            if item.name not in names:
                names.append(item.name)
    return names


def _speedup_cell(item: DefinitionProgress | None) -> str:
    if item is None:
        return "待测"
    if item.state == "FAILED":
        return "—"
    if item.best_geo_mean is None:
        return item.state
    suffix = "" if item.state == "PASSED" else "*"
    return f"{item.best_geo_mean:.6g}x{suffix}"


def render_markdown(reports: list[DeviceReport]) -> str:
    lines = [
        "| 设备 | 通过 | 失败 | >1x | 总数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report.name} | {report.passed} | {report.failed} | "
            f"{report.accelerated} | {len(report.definitions)} |"
        )

    lines.extend([
        "",
        "| 算子 | " + " | ".join(report.name for report in reports) + " |",
        "|---|" + "|".join("---:" for _ in reports) + "|",
    ])
    by_device = {
        report.name: {item.name: item for item in report.definitions}
        for report in reports
    }
    for definition_name in definition_names(reports):
        cells = [
            _speedup_cell(by_device[report.name].get(definition_name))
            for report in reports
        ]
        lines.append(f"| `{definition_name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def report_json(reports: list[DeviceReport]) -> dict:
    return {
        "devices": [
            {
                "name": report.name,
                "workspace": str(report.workspace),
                "passed": report.passed,
                "failed": report.failed,
                "accelerated": report.accelerated,
                "definitions": [asdict(item) for item in report.definitions],
            }
            for report in reports
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        metavar="DEVICE=PATH",
        help="add or override one workspace",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    reports = discover_reports(root, args.run_name) if root.is_dir() else []
    by_name = {report.name: report for report in reports}
    for value in args.workspace:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("--workspace must use DEVICE=PATH")
        by_name[name] = load_report(name, Path(path).expanduser())

    names = list(DEFAULT_DEVICE_ORDER)
    names.extend(sorted(name for name in by_name if name not in names))
    reports = [by_name[name] for name in names if name in by_name]
    if not reports:
        parser.error("no collected workspaces found")

    markdown = render_markdown(reports)
    print(markdown, end="")
    if args.markdown_out:
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(report_json(reports), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
