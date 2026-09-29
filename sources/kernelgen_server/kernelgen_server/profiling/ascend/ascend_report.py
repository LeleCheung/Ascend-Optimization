# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Plain-text rendering for Huawei Ascend msprof exports.

The report is deliberately factual. It formats vendor-exported tables and
collection metadata for direct reading by an agent, but does not diagnose
bottlenecks or suggest optimizations.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..models import ProfileOptions
from ..source_view import ReportSourceTarget


_OP_STATISTIC_COLUMNS = [
    "Device_id",
    "OP Type",
    "Core Type",
    "Count",
    "Total Time(us)",
    "Min Time(us)",
    "Avg Time(us)",
    "Max Time(us)",
    "Ratio(%)",
]
_API_STATISTIC_COLUMNS = [
    "Device_id",
    "Level",
    "API Name",
    "Time(us)",
    "Count",
    "Avg(us)",
    "Min(us)",
    "Max(us)",
    "Variance",
]
_TASK_TIME_COLUMNS = [
    "Device_id",
    "kernel_name",
    "kernel_type",
    "stream_id",
    "task_id",
    "task_time(us)",
    "task_start(us)",
    "task_stop(us)",
]
_OPERATION_METRICS = [
    ("aicore_time_us", "us"),
    ("aic_total_cycles", "cycle"),
    ("aic_mac_time_us", "us"),
    ("aic_mac_ratio", "ratio"),
    ("aic_scalar_time_us", "us"),
    ("aic_scalar_ratio", "ratio"),
    ("aic_mte1_time_us", "us"),
    ("aic_mte1_ratio", "ratio"),
    ("aic_mte2_time_us", "us"),
    ("aic_mte2_ratio", "ratio"),
    ("aic_fixpipe_time_us", "us"),
    ("aic_fixpipe_ratio", "ratio"),
    ("aic_icache_miss_rate", "ratio"),
    ("aiv_time_us", "us"),
    ("aiv_total_cycles", "cycle"),
    ("aiv_vec_time_us", "us"),
    ("aiv_vec_ratio", "ratio"),
    ("aiv_scalar_time_us", "us"),
    ("aiv_scalar_ratio", "ratio"),
    ("aiv_mte2_time_us", "us"),
    ("aiv_mte2_ratio", "ratio"),
    ("aiv_mte3_time_us", "us"),
    ("aiv_mte3_ratio", "ratio"),
    ("aiv_icache_miss_rate", "ratio"),
    ("cube_utilization_pct", "%"),
]
_OP_DETAIL_METRIC_ORDER = [
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryUB",
    "MemoryL0",
    "L2Cache",
    "ResourceConflictRatio",
    "MemoryDetail",
    "Occupancy",
    "Roofline",
]
_OP_DETAIL_BASIC_COLUMNS = [
    "Op Name",
    "Op Type",
    "Task Duration(us)",
    "Block Dim",
    "Mix Block Dim",
    "Device Id",
    "Current Freq",
    "Rated Freq",
]


def _first_matching(root: Path, pattern: str) -> Optional[Path]:
    return next(iter(root.rglob(pattern)), None)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _select_columns(rows: Sequence[Dict[str, str]], preferred: Sequence[str]) -> List[str]:
    if not rows:
        return []
    available = set(rows[0])
    selected = [column for column in preferred if column in available]
    return selected or list(rows[0])


def _render_table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    normalized = [[_clean(value) for value in row] for row in rows]
    if not columns:
        return ["    No columns were exported."]
    widths = [len(_clean(column)) for column in columns]
    for row in normalized:
        for index in range(min(len(row), len(widths))):
            widths[index] = max(widths[index], len(row[index]))

    def render_row(row: Sequence[str]) -> str:
        padded = [row[index].ljust(widths[index]) for index in range(len(widths))]
        return "    " + "  ".join(padded).rstrip()

    header = [_clean(column) for column in columns]
    lines = [render_row(header), "    " + "  ".join("-" * width for width in widths)]
    lines.extend(render_row(row + [""] * (len(widths) - len(row))) for row in normalized)
    return lines


def _append_csv_section(
    lines: List[str], title: str, path: Optional[Path], preferred_columns: Sequence[str]
) -> None:
    lines.extend([f"Section: {title}", ""])
    if path is None:
        lines.extend(["    Not exported by msprof.", ""])
        return
    try:
        rows = _read_csv(path)
    except (OSError, csv.Error, UnicodeError) as exc:
        lines.extend([f"    Could not read {path.name}: {_clean(exc)}", ""])
        return
    lines.append(f"    Source File: {path.name}")
    lines.append(f"    Row Count: {len(rows)}")
    lines.append("")
    if rows:
        columns = _select_columns(rows, preferred_columns)
        lines.extend(
            _render_table(columns, ([row.get(column, "") for column in columns] for row in rows))
        )
    else:
        lines.append("    No rows were exported.")
    lines.append("")


def _append_measured_operations(
    lines: List[str], parsed: Optional[Dict[str, Any]], warmup: int, iterations: int
) -> None:
    lines.extend(["Section: Measured Operator Details", ""])
    lines.append(f"    Aggregation: exclude the first {warmup} matching invocation(s) per operator")
    lines.append(f"    Requested measured invocations: {iterations}")
    lines.append("")
    operations = parsed.get("ops", []) if parsed else []
    if not operations:
        lines.extend(["    No measured operator details were available.", ""])
        return

    for index, operation in enumerate(operations, start=1):
        lines.append(f"    [{index}] {_clean(operation.get('op_name'))}")
        fields = [
            ("OP Type", operation.get("op_type", "")),
            ("Task Type", operation.get("task_type", "")),
            ("Block Dim", operation.get("block_dim", "")),
            ("Measured Invocations", operation.get("measured_invocations", "")),
            ("Total Duration", operation.get("total_duration_us", "")),
            ("Average Duration", operation.get("avg_duration_us", "")),
            ("Minimum Duration", operation.get("min_duration_us", "")),
            ("Maximum Duration", operation.get("max_duration_us", "")),
            ("Average Wait Time", operation.get("avg_wait_time_us", "")),
            ("Time Unit", "us"),
            ("Input Shapes", operation.get("input_shapes", "")),
            ("Input Data Types", operation.get("input_dtypes", "")),
        ]
        lines.extend(_render_table(["Field", "Value"], fields))
        metric_rows = [
            (metric, unit, operation[metric])
            for metric, unit in _OPERATION_METRICS
            if metric in operation
        ]
        if metric_rows:
            lines.append("")
            lines.extend(_render_table(["Metric Name", "Metric Unit", "Metric Value"], metric_rows))
        lines.append("")


def _metric_group_name(path: Path) -> str:
    return re.sub(r"_\d{14,}$", "", path.stem)


def _metric_unit(column: str) -> str:
    match = re.search(r"\(([^()]*)\)(?:\(estimate\))?$", column)
    if match:
        return match.group(1)
    lowered = column.lower()
    if "cycles" in lowered:
        return "cycle"
    if "instructions" in lowered or "instr_number" in lowered:
        return "instruction"
    if lowered.endswith("_fops"):
        return "operation"
    if "ratio" in lowered or "rate" in lowered:
        return "ratio"
    if any(token in lowered for token in ("cache_hit", "miss_allocate")):
        return "count"
    return "value"


def _append_op_basic_info(lines: List[str], root: Path, kernel_name: str) -> None:
    paths = sorted(root.rglob("OpBasicInfo*.csv"))
    lines.extend(["Section: msprof op - Operator Basic Information", ""])
    lines.append(f"    Kernel: {kernel_name}")
    if not paths:
        lines.extend(["    Not exported by msprof op.", ""])
        return
    rows: List[Dict[str, str]] = []
    for path in paths:
        try:
            rows.extend(_read_csv(path))
        except (OSError, csv.Error, UnicodeError):
            continue
    lines.append(f"    Source Files: {len(paths)}")
    lines.append(f"    Launch Records: {len(rows)}")
    lines.append("")
    if not rows:
        lines.extend(["    No operator basic information rows were readable.", ""])
        return
    columns = _select_columns(rows, _OP_DETAIL_BASIC_COLUMNS)
    lines.extend(
        _render_table(columns, ([row.get(column, "") for column in columns] for row in rows))
    )
    lines.append("")


def _append_op_metric_group(
    lines: List[str], kernel_name: str, name: str, paths: Sequence[Path]
) -> None:
    lines.extend([f"Section: msprof op - {name}", ""])
    lines.append(f"    Kernel: {kernel_name}")
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    row_count = 0
    for path in paths:
        try:
            rows = _read_csv(path)
        except (OSError, csv.Error, UnicodeError):
            continue
        row_count += len(rows)
        for row in rows:
            sub_block = _clean(row.get("sub_block_id")) or "all"
            for column, raw_value in row.items():
                if column in {"block_id", "sub_block_id"}:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                grouped[sub_block][column].append(value)

    lines.append(f"    Source Files: {len(paths)}")
    lines.append(f"    Source Rows: {row_count}")
    lines.append("")
    if not grouped:
        lines.extend(["    No numeric metric rows were readable.", ""])
        return
    for sub_block, metrics in sorted(grouped.items()):
        lines.append(f"    Sub-block: {sub_block}")
        rows = []
        for metric, values in metrics.items():
            rows.append(
                (
                    metric,
                    _metric_unit(metric),
                    len(values),
                    f"{min(values):.6f}",
                    f"{sum(values) / len(values):.6f}",
                    f"{max(values):.6f}",
                )
            )
        lines.extend(
            _render_table(
                ["Metric Name", "Metric Unit", "Samples", "Minimum", "Average", "Maximum"], rows
            )
        )
        lines.append("")


def _append_op_detailed_profile(
    lines: List[str], root: Optional[Path], kernel_names: Sequence[str], metrics: str
) -> None:
    lines.extend(["Section: msprof op Detailed Kernel Profile", ""])
    if root is None or not root.is_dir():
        lines.extend(["    Not collected.", ""])
        return
    lines.append(f"    Kernel Count: {len(kernel_names)}")
    lines.append(f"    Kernels: {', '.join(kernel_names) if kernel_names else 'unknown'}")
    lines.append(f"    Requested Metrics: {metrics}")
    lines.append("")
    for index, kernel_name in enumerate(kernel_names):
        kernel_root = root / f"kernel-{index:03d}"
        if not kernel_root.is_dir():
            continue
        lines.extend(
            [
                f"Section: msprof op Kernel {index + 1}",
                "",
                f"    Kernel: {kernel_name}",
                f"    Report Directory: {kernel_root.name}",
                "",
            ]
        )
        _append_op_basic_info(lines, kernel_root, kernel_name)

        by_name: Dict[str, List[Path]] = defaultdict(list)
        for path in kernel_root.rglob("*.csv"):
            name = _metric_group_name(path)
            if name != "OpBasicInfo":
                by_name[name].append(path)
        ordered = [name for name in _OP_DETAIL_METRIC_ORDER if name in by_name]
        ordered.extend(sorted(name for name in by_name if name not in ordered))
        for name in ordered:
            _append_op_metric_group(lines, kernel_name, name, sorted(by_name[name]))


def _append_timeline_summary(lines: List[str], path: Optional[Path]) -> None:
    lines.extend(["Section: Timeline Summary", ""])
    if path is None:
        lines.extend(["    Not exported by msprof.", ""])
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        lines.extend([f"    Could not read {path.name}: {_clean(exc)}", ""])
        return
    events = payload if isinstance(payload, list) else payload.get("traceEvents", [])
    if not isinstance(events, list):
        events = []

    grouped: Dict[tuple[str, str, str], Dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "duration": 0.0}
    )
    complete_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        name = _clean(event.get("name"))
        category = _clean(event.get("cat"))
        phase = _clean(event.get("ph"))
        entry = grouped[(name, category, phase)]
        entry["count"] += 1
        try:
            entry["duration"] += float(event.get("dur", 0) or 0)
        except (TypeError, ValueError):
            pass
        if phase == "X":
            complete_count += 1

    lines.append(f"    Source File: {path.name}")
    lines.append(f"    Event Count: {len(events)}")
    lines.append(f"    Complete Events: {complete_count}")
    lines.append("")
    if grouped:
        rows = [
            (name, category, phase, int(values["count"]), f"{values['duration']:.6f}")
            for (name, category, phase), values in sorted(grouped.items())
        ]
        lines.extend(
            _render_table(["Event Name", "Category", "Phase", "Count", "Total Duration(us)"], rows)
        )
    else:
        lines.append("    No timeline events were exported.")
    lines.append("")


def _append_exported_files(lines: List[str], root: Path) -> None:
    lines.extend(["Section: Exported Files", ""])
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [(str(path.relative_to(root)), path.stat().st_size) for path in files]
    lines.extend(_render_table(["Relative Path", "Size (bytes)"], rows))
    lines.append("")


def render_msprof_details(
    profile_root: Path,
    target: ReportSourceTarget,
    options: ProfileOptions,
    device: str,
    parsed_operations: Optional[Dict[str, Any]] = None,
    op_profile_root: Optional[Path] = None,
    op_kernel_names: Sequence[str] = (),
    op_metrics: str = "",
) -> str:
    """Render one msprof export as a complete, factual plain-text report."""
    backend_options = options.backend_options.get("npu", {})
    aic_metrics = backend_options.get("aic_metrics", "PipeUtilization")
    workload_inputs = json.dumps(
        target.workload.inputs,
        sort_keys=True,
        ensure_ascii=False,
    )
    lines = [
        "==PROF== Huawei Ascend msprof Profile",
        f"==PROF== Definition: {target.definition.name}",
        f"==PROF== Implementation: {target.implementation.name}",
        f"==PROF== Workload: {target.workload.name}",
        f"==PROF== Device: {device}",
        "",
        "Section: Profile Overview",
        "",
    ]
    overview = [
        ("Definition", target.definition.name),
        ("Implementation", target.implementation.name),
        ("Workload", target.workload.name),
        ("Workload Inputs", workload_inputs),
        ("Device", device),
        ("Profile Level", options.level),
        ("Warmup Invocations", options.warmup),
        ("Measured Invocations", options.iterations),
        ("AI Core Collection", "on" if backend_options.get("ai_core", True) else "off"),
        ("AI Core Metrics", aic_metrics or "disabled"),
    ]
    lines.extend(_render_table(["Field", "Value"], overview))
    lines.append("")

    _append_measured_operations(lines, parsed_operations, options.warmup, options.iterations)
    _append_op_detailed_profile(lines, op_profile_root, op_kernel_names, op_metrics)
    _append_csv_section(
        lines,
        "Operator Statistics",
        _first_matching(profile_root, "op_statistic_*.csv"),
        _OP_STATISTIC_COLUMNS,
    )
    _append_csv_section(
        lines,
        "Host API Statistics",
        _first_matching(profile_root, "api_statistic_*.csv"),
        _API_STATISTIC_COLUMNS,
    )
    _append_csv_section(
        lines, "Task Time", _first_matching(profile_root, "task_time_*.csv"), _TASK_TIME_COLUMNS
    )
    timeline = _first_matching(profile_root, "msprof_*.json")
    if timeline is None:
        timeline = _first_matching(profile_root, "trace.json")
    _append_timeline_summary(lines, timeline)
    _append_exported_files(lines, profile_root)
    return "\n".join(lines).rstrip() + "\n"
