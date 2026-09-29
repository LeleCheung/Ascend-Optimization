# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Normalize Moore Threads MCU metric reports for agents and clients."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..models import ProfileOptions
from ..source_view import ReportSourceTarget


class McuReportError(RuntimeError):
    """Raised when MCU output does not contain usable metric evidence."""


@dataclass(frozen=True)
class McuAnalysis:
    metrics: dict[str, Any]


_PROGRESS_RE = re.compile(
    r'^==PROF== Profiling "(?P<name>.*?)" - (?P<launch>\d+): '
    r"Application replay pass (?P<replay>\d+)\s*$"
)
_KERNEL_RE = re.compile(
    r"^\s{2}(?P<name>.+?)\s+"
    r"\((?P<grid>\d+\s*,\s*\d+\s*,\s*\d+)\)x"
    r"\((?P<block>\d+\s*,\s*\d+\s*,\s*\d+)\),\s*"
    r"Context\s+(?P<context>\d+),\s*"
    r"Stream\s+(?P<stream>\d+),\s*"
    r"Device\s+(?P<device>\d+)\s*$"
)
_SECTION_RE = re.compile(r"^\s*Section:\s*(?P<name>.+?)\s*$")
_DASH_RE = re.compile(r"-{3,}")


def _triplet(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",")]


def _numeric_value(raw_value: str) -> int | float | None:
    value = raw_value.strip().replace(",", "")
    if not value or value.lower() in {"n/a", "na", "nan", "inf", "-inf"}:
        return None
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _separator_spans(line: str) -> list[tuple[int, int]]:
    return [match.span() for match in _DASH_RE.finditer(line)]


def _parse_metric_table(
    lines: Sequence[str], start: int
) -> tuple[list[dict[str, Any]], int]:
    if start + 2 >= len(lines):
        return [], start
    spans = _separator_spans(lines[start])
    if len(spans) < 3 or len(_separator_spans(lines[start + 2])) < 3:
        return [], start

    rows: list[dict[str, Any]] = []
    index = start + 3
    while index < len(lines):
        line = lines[index]
        if _separator_spans(line):
            return rows, index + 1
        if not line.strip():
            return rows, index + 1
        cells = [line[column_start:column_end].strip() for column_start, column_end in spans]
        if len(cells) >= 3 and cells[0]:
            raw_value = cells[2]
            rows.append(
                {
                    "name": cells[0],
                    "unit": cells[1],
                    "raw_value": raw_value,
                    "value": _numeric_value(raw_value),
                }
            )
        index += 1
    return rows, index


def _metric_values(
    kernel: dict[str, Any], name: str, unit: str | None = None
) -> list[float]:
    values: list[float] = []
    for section in kernel.get("sections", []):
        for metric in section.get("metrics", []):
            value = metric.get("value")
            if metric.get("name") != name or not isinstance(value, (int, float)):
                continue
            if unit is not None and metric.get("unit") != unit:
                continue
            values.append(float(value))
    return values


def _duration_us(kernel: dict[str, Any]) -> float | None:
    for unit, multiplier in (("us", 1.0), ("ns", 0.001), ("ms", 1000.0), ("s", 1_000_000.0)):
        values = _metric_values(kernel, "Duration", unit)
        if values:
            return values[0] * multiplier
    return None


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else None


def _aggregate_kernels(kernels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for kernel in kernels:
        grouped.setdefault(str(kernel["name"]), []).append(kernel)

    summaries: list[dict[str, Any]] = []
    for name, invocations in grouped.items():
        summary: dict[str, Any] = {
            "name": name,
            "invocation_count": len(invocations),
            "grid": invocations[0]["grid"],
            "block": invocations[0]["block"],
            "device": invocations[0]["device"],
        }
        durations = [value for item in invocations if (value := _duration_us(item)) is not None]
        if durations:
            summary.update(
                {
                    "total_duration_us": round(sum(durations), 6),
                    "avg_duration_us": _mean(durations),
                    "min_duration_us": round(min(durations), 6),
                    "max_duration_us": round(max(durations), 6),
                }
            )

        metric_specs = {
            "compute_throughput_pct": ("Compute(MP) Throughput", "%"),
            "memory_throughput_pct": ("Memory Throughput", "%"),
            "l1_hit_rate_pct": ("L1 Hit Rate", "%"),
            "l2_hit_rate_pct": ("L2 Hit Rate", "%"),
            "llc_hit_rate_pct": ("LLC Hit Rate", "%"),
            "waves_per_mp": ("Waves Per MP", ""),
            "registers_per_thread": ("Registers Per Thread", "register/thread"),
        }
        for field, (metric_name, unit) in metric_specs.items():
            values = [
                value
                for invocation in invocations
                for value in _metric_values(invocation, metric_name, unit)
            ]
            average = _mean(values)
            if average is not None:
                summary[f"avg_{field}"] = average
                summary[f"max_{field}"] = round(max(values), 6)
        summaries.append(summary)
    return summaries


def analyze_mcu_output(output: str) -> McuAnalysis:
    """Parse MCU 1.2+ details output into stable JSON-compatible metrics."""
    lines = output.splitlines()
    replay_passes: set[int] = set()
    launch_ids: set[int] = set()
    kernels: list[dict[str, Any]] = []
    current_kernel: dict[str, Any] | None = None

    index = 0
    while index < len(lines):
        line = lines[index]
        progress = _PROGRESS_RE.match(line)
        if progress:
            replay_passes.add(int(progress.group("replay")))
            launch_ids.add(int(progress.group("launch")))
            index += 1
            continue

        kernel_match = _KERNEL_RE.match(line)
        if kernel_match:
            current_kernel = {
                "index": len(kernels),
                "name": kernel_match.group("name"),
                "grid": _triplet(kernel_match.group("grid")),
                "block": _triplet(kernel_match.group("block")),
                "context": int(kernel_match.group("context")),
                "stream": int(kernel_match.group("stream")),
                "device": int(kernel_match.group("device")),
                "sections": [],
            }
            kernels.append(current_kernel)
            index += 1
            continue

        section_match = _SECTION_RE.match(line)
        if section_match and current_kernel is not None:
            metrics, next_index = _parse_metric_table(lines, index + 1)
            if metrics:
                current_kernel["sections"].append(
                    {"name": section_match.group("name"), "metrics": metrics}
                )
                index = next_index
                continue
        index += 1

    metric_row_count = sum(
        len(section["metrics"])
        for kernel in kernels
        for section in kernel["sections"]
    )
    if not kernels or metric_row_count == 0:
        raise McuReportError("MCU output did not contain kernel metric tables")

    summaries = _aggregate_kernels(kernels)
    metrics = {
        "format_version": 1,
        "profiler": "mcu",
        "application_replay_passes": len(replay_passes),
        "profiled_launch_count": len(launch_ids),
        "kernel_invocation_count": len(kernels),
        "unique_kernel_count": len(summaries),
        "metric_row_count": metric_row_count,
        "kernel_summaries": summaries,
        "kernels": kernels,
    }
    return McuAnalysis(metrics=metrics)


def _clean(value: Any) -> str:
    return " ".join(str(value).replace("\x00", " ").split())


def _shorten(value: Any, limit: int = 100) -> str:
    text = _clean(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _display(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return _clean(value)


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    rendered = [[_display(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for column, value in enumerate(row):
            widths[column] = max(widths[column], len(value))
    lines = [
        "    " + "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "    " + "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "    " + "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rendered
    )
    return lines


def render_mcu_details(
    analysis: McuAnalysis,
    target: ReportSourceTarget,
    options: ProfileOptions,
    device: str,
    executable: str,
) -> str:
    """Render a compact three-section report without duplicating artifact metadata."""
    metrics = analysis.metrics
    lines = [
        "Section: Overview",
        "",
        f"    Backend: {target.expected_backend}",
        "    Profiler: mcu",
        f"    Level: {options.level}",
        f"    Definition: {target.definition.name}",
        f"    Implementation: {target.implementation.name}",
        f"    Workload: {target.workload.name}",
        f"    Assigned Device: {device}",
        "    Runner Device After Visibility Binding: musa:0",
        f"    MCU Executable: {executable}",
        f"    Warmup Iterations: {options.warmup}",
        f"    Profiled Iterations: {options.iterations}",
        f"    Application Replay Passes: {metrics['application_replay_passes']}",
        "    Timing Note: MCU instrumented duration is diagnostic, not evaluator latency.",
        "",
        "Section: Kernel Summary",
        "",
        f"    Kernel Invocations: {metrics['kernel_invocation_count']}",
        f"    Unique Kernels: {metrics['unique_kernel_count']}",
        f"    Metric Rows: {metrics['metric_row_count']}",
        "",
    ]

    summary_rows = []
    for summary in metrics["kernel_summaries"]:
        summary_rows.append(
            (
                _shorten(summary["name"]),
                summary["invocation_count"],
                "x".join(str(value) for value in summary["grid"]),
                "x".join(str(value) for value in summary["block"]),
                summary.get("avg_duration_us"),
                summary.get("max_duration_us"),
            )
        )
    lines.extend(
        _render_table(
            ["Kernel", "Calls", "Grid", "Block", "Avg us", "Max us"],
            summary_rows,
        )
        if summary_rows
        else ["    No kernel summaries were available."]
    )
    lines.extend(["", "Section: Performance Signals", ""])

    signal_rows = []
    for summary in metrics["kernel_summaries"]:
        signal_rows.append(
            (
                _shorten(summary["name"], 72),
                summary.get("avg_compute_throughput_pct"),
                summary.get("max_compute_throughput_pct"),
                summary.get("avg_memory_throughput_pct"),
                summary.get("avg_l1_hit_rate_pct"),
                summary.get("avg_l2_hit_rate_pct"),
                summary.get("avg_waves_per_mp"),
                summary.get("avg_registers_per_thread"),
            )
        )
    lines.extend(
        _render_table(
            [
                "Kernel",
                "Compute Avg %",
                "Compute Peak %",
                "Memory Avg %",
                "L1 Hit %",
                "L2 Hit %",
                "Waves/MP",
                "Regs/Thread",
            ],
            signal_rows,
        )
        if signal_rows
        else ["    No normalized performance signals were available."]
    )
    lines.append("")
    return "\n".join(lines)
