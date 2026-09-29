# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""XProfiler trace parsing and agent-facing report rendering."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


_TRACE_CATEGORIES = {
    "api": "TaskSystemApi",
    "memcpy": "TaskSystemMemcpy",
    "wait": "TaskSystemWait",
}


def _as_number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float) -> float:
    return round(value, 3)


def _clean(value: Any) -> str:
    return str(value if value is not None else "").replace("\n", " ").strip()


def _render_table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    normalized = [[_clean(value) for value in row] for row in rows]
    widths = [len(column) for column in columns]
    for row in normalized:
        for index, value in enumerate(row[: len(widths)]):
            widths[index] = max(widths[index], len(value))

    def line(row: Sequence[str]) -> str:
        values = list(row) + [""] * (len(widths) - len(row))
        return (
            "    "
            + "  ".join(
                values[index].ljust(widths[index]) for index in range(len(widths))
            ).rstrip()
        )

    output = [line(columns), "    " + "  ".join("-" * width for width in widths)]
    output.extend(line(row) for row in normalized)
    return output


def _render_xprofiler_details(
    metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    device: str,
    warmup: int,
    iterations: int,
) -> str:
    """Render the normalized, agent-facing XProfiler report."""
    lines = [
        "Kunlunxin XProfiler Metrics Summary",
        "====================================",
        "",
        "Section: Overview",
        "",
    ]
    lines.extend(
        _render_table(
            ["Field", "Value"],
            [
                ("Device", device),
                ("Warmup Invocations", warmup),
                ("Measured Invocations", iterations),
                ("Kernel Invocations", summary.get("kernel_invocation_count", 0)),
                ("Distinct Kernel Operations", summary.get("operation_count", 0)),
                ("Average Kernel Time / Call (us)", summary.get("avg_time_us", "")),
                ("Average Wall Time / Call (us)", summary.get("avg_wall_time_us", "")),
                ("Trace Complete", summary.get("trace_complete", False)),
                (
                    "Timing Authority",
                    "diagnostic only; evaluator latency is authoritative",
                ),
            ],
        )
    )
    lines.extend(["", "Section: Kernel Summary", ""])
    operations = metrics.get("ops", [])
    if operations:
        lines.extend(
            _render_table(
                [
                    "Kernel",
                    "Calls",
                    "Total (us)",
                    "Average (us)",
                    "Minimum (us)",
                    "Maximum (us)",
                    "Clusters",
                    "Cores",
                ],
                (
                    (
                        operation.get("op_name", ""),
                        operation.get("measured_invocations", ""),
                        operation.get("total_duration_us", ""),
                        operation.get("avg_duration_us", ""),
                        operation.get("min_duration_us", ""),
                        operation.get("max_duration_us", ""),
                        operation.get("nclusters", ""),
                        operation.get("ncores", ""),
                    )
                    for operation in operations
                ),
            )
        )
    else:
        lines.append("    No measured kernels were parsed.")

    lines.extend(["", "Section: Performance Signals", ""])
    signal_rows = []
    for name in ("api", "memcpy", "wait"):
        signal = metrics.get(name, {})
        signal_rows.append(
            (
                name,
                signal.get("count", 0),
                signal.get("total_duration_us", 0),
                signal.get("total_bytes", 0),
            )
        )
    lines.extend(
        _render_table(
            ["Signal", "Events", "Total Duration (us)", "Total Bytes"], signal_rows
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def _aggregate_events(
    events: List[Dict[str, Any]],
    category: str,
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> Dict[str, Any]:
    """Aggregate complete Chrome trace events by name and measurement window."""
    grouped: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "total_duration_us": 0.0, "total_bytes": 0.0}
    )
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != category:
            continue
        duration_us = max(_as_number(event.get("dur")), 0.0)
        if start_ns is not None and end_ns is not None:
            timestamp_us = _as_number(event.get("ts"), -1.0)
            if (
                timestamp_us < start_ns / 1000.0
                or timestamp_us + duration_us > end_ns / 1000.0
            ):
                continue
        name = str(event.get("name") or "unknown")
        item = grouped[name]
        item["count"] += 1
        item["total_duration_us"] += duration_us
        args = event.get("args")
        if isinstance(args, dict):
            item["total_bytes"] += max(_as_number(args.get("size (byte)")), 0.0)

    operations = []
    for name, values in grouped.items():
        count = int(values["count"])
        detail: Dict[str, Any] = {
            "name": name,
            "count": count,
            "total_duration_us": _round(values["total_duration_us"]),
            "avg_duration_us": _round(values["total_duration_us"] / count),
        }
        if values["total_bytes"]:
            detail["total_bytes"] = int(values["total_bytes"])
        operations.append(detail)
    operations.sort(key=lambda item: item["total_duration_us"], reverse=True)
    return {
        "count": sum(item["count"] for item in operations),
        "total_duration_us": _round(
            sum(item["total_duration_us"] for item in operations)
        ),
        "total_bytes": sum(item.get("total_bytes", 0) for item in operations),
        "operations": operations,
    }


def _parse_xprofiler_report(
    path: Path,
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> Dict[str, Any]:
    """Normalize XProfiler's Chrome trace into the common metrics envelope."""
    if (start_ns is None) != (end_ns is None):
        raise ValueError(
            "XProfiler measurement window requires both start_ns and end_ns"
        )
    if start_ns is not None and end_ns is not None and end_ns <= start_ns:
        raise ValueError("XProfiler measurement window end must be after start")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid XProfiler Chrome trace: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("traceEvents"), list
    ):
        raise ValueError("XProfiler report is missing a traceEvents list")
    events = [event for event in payload["traceEvents"] if isinstance(event, dict)]

    # TaskSSEduration is the device execution timeline. Kernel entries carry the
    # real kernel name in args.name; async_dma and TaskSystemKernel/flush are not
    # candidate kernels and must not contribute to kernel latency.
    kernels = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "TaskSSEduration":
            continue
        args = event.get("args")
        if not isinstance(args, dict) or not args.get("name"):
            continue
        duration_us = _as_number(event.get("dur"), -1.0)
        if duration_us < 0:
            continue
        timestamp_us = _as_number(event.get("ts"), -1.0)
        if start_ns is not None and end_ns is not None:
            start_us = start_ns / 1000.0
            end_us = end_ns / 1000.0
            if timestamp_us < start_us or timestamp_us + duration_us > end_us:
                continue
        kernels.append(
            {
                "name": str(args["name"]),
                "duration_us": _round(duration_us),
                "device": args.get("device"),
                "channel": args.get("channel"),
                "code_addr": str(args.get("code_addr", "")),
                "nclusters": int(_as_number(args.get("nclusters"))),
                "ncores": int(_as_number(args.get("ncores"))),
            }
        )
    if not kernels:
        raise ValueError("XProfiler trace contains no device kernel events")

    grouped_kernels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for kernel in kernels:
        grouped_kernels[kernel["name"]].append(kernel)
    operations = []
    for name, samples in grouped_kernels.items():
        durations = [sample["duration_us"] for sample in samples]
        operations.append(
            {
                "op_name": name,
                "measured_invocations": len(samples),
                "total_duration_us": _round(sum(durations)),
                "avg_duration_us": _round(sum(durations) / len(durations)),
                "min_duration_us": _round(min(durations)),
                "max_duration_us": _round(max(durations)),
                "code_addr": samples[0]["code_addr"],
                "nclusters": samples[0]["nclusters"],
                "ncores": samples[0]["ncores"],
            }
        )
    operations.sort(key=lambda item: item["total_duration_us"], reverse=True)
    kernel_time_us = _round(sum(kernel["duration_us"] for kernel in kernels))

    metrics: Dict[str, Any] = {
        "kernel_time_us": kernel_time_us,
        "kernels": kernels,
        "ops": operations,
    }
    for key, category in _TRACE_CATEGORIES.items():
        metrics[key] = _aggregate_events(
            events, category, start_ns=start_ns, end_ns=end_ns
        )
    return metrics
