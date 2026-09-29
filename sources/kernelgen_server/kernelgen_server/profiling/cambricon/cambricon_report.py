# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""CNPerf, PMU, MLISA, and source-mapping report normalization."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..models import ProfileOptions
from ..source_view import ReportSourceTarget


_IR_LOCATION_RE = re.compile(
    r'^#(?P<id>loc\d*)\s*=\s*loc\("(?P<path>[^"]+)":'
    r"(?P<line>\d+):(?P<column>\d+)\)"
)
_IR_LOCATION_REF_RE = re.compile(r"loc\(#(?P<id>loc\d*)\)")
_IR_DIRECT_LOCATION_RE = re.compile(
    r'loc\("(?P<path>[^"]+)":(?P<line>\d+):(?P<column>\d+)\)'
)
_IR_FUNCTION_RE = re.compile(
    r"\b(?:tt|mlu|func)\.func\b.*?@(?P<name>[A-Za-z_.$][\w.$-]*)"
)
_MLISA_FUNCTION_RE = re.compile(
    r"^\.visible\s+\.kernel\s+(?P<name>[A-Za-z_.$][\w.$-]*)\s*\("
)
_COUNTER_HEADER_RE = re.compile(r"^(?P<name>.+)\((?P<unit>[^()]*)\)$")
_COMMON_PMU_COLUMNS = {
    "Kernel Name",
    "Dim",
    "TID",
    "Duration(ns)",
    "Visible Cluster",
    "Device ID",
}
_IR_FORMATS = {
    ".ttir": "ttir",
    ".linalg": "linalg-ir",
    ".linalgopt": "linalg-opt-ir",
    ".mluir": "mluir",
    ".mluiropt": "mluir-opt",
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default


def _normalize_source_path(path: str, target: ReportSourceTarget) -> str:
    normalized = path.replace("\\", "/")
    sources = [source.path.replace("\\", "/") for source in target.implementation.sources]
    exact = [
        source
        for source in sources
        if normalized == source or normalized.endswith(f"/{source}")
    ]
    if exact:
        return exact[0]
    basename = Path(normalized).name
    matches = [source for source in sources if Path(source).name == basename]
    return matches[0] if len(matches) == 1 else normalized


def parse_kernel_statistics(path: Path) -> Dict[str, Any]:
    """Parse CNPerf's report-mode KernelStatistics.csv (durations are ns)."""
    kernels: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Total", "Avg", "Min", "Max", "Occurrence", "Name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CNPerf kernel statistics did not contain expected columns")
        for row in reader:
            name = str(row.get("Name", "")).strip().strip('"')
            if not name or name == "[total]":
                continue
            calls = max(0, int(_number(row.get("Occurrence"))))
            kernels.append(
                {
                    "name": name,
                    "time_percent": _number(row.get("Ratio(%)")),
                    "calls": calls,
                    "total_duration_us": _number(row.get("Total")) / 1_000.0,
                    "average_duration_us": _number(row.get("Avg")) / 1_000.0,
                    "minimum_duration_us": _number(row.get("Min")) / 1_000.0,
                    "maximum_duration_us": _number(row.get("Max")) / 1_000.0,
                    "p90_duration_us": _number(row.get("Pct90")) / 1_000.0,
                    "standard_deviation_us": _number(row.get("StdDev")) / 1_000.0,
                }
            )
    if not kernels:
        raise ValueError("CNPerf kernel statistics did not contain any kernels")
    kernels.sort(key=lambda item: item["total_duration_us"], reverse=True)
    return {
        "time_unit": "microseconds",
        "kernel_count": len(kernels),
        "kernel_invocation_count": sum(item["calls"] for item in kernels),
        "total_kernel_time_us": sum(item["total_duration_us"] for item in kernels),
        "kernels": kernels,
    }


def _event_timestamp(event: Dict[str, Any]) -> float:
    return _number(event.get("ts"))


def _capture_window(events: Iterable[Dict[str, Any]]) -> tuple[float, float]:
    ordered = sorted(events, key=_event_timestamp)
    starts = [
        event
        for event in ordered
        if event.get("name") in {"cnProfilerStart", "cnrtProfilerStart"}
    ]
    stops = [
        event
        for event in ordered
        if event.get("name") in {"cnProfilerStop", "cnrtProfilerStop"}
    ]
    for start in starts:
        start_us = _event_timestamp(start) + _number(start.get("dur"))
        stop = next(
            (event for event in stops if _event_timestamp(event) >= start_us),
            None,
        )
        if stop is not None:
            return start_us, _event_timestamp(stop)
    raise ValueError("CNPerf timeline did not contain a complete profiler start/stop range")


def parse_cnperf_timeline(path: Path) -> Dict[str, Any]:
    """Normalize device kernels inside the cnProfilerApi capture range."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(raw_events, list):
        raise ValueError("CNPerf timechart did not contain a traceEvents list")
    events = [event for event in raw_events if isinstance(event, dict)]
    start_us, stop_us = _capture_window(events)

    timeline: List[Dict[str, Any]] = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        timestamp = _event_timestamp(event)
        duration = _number(event.get("dur"))
        if timestamp < start_us or timestamp + duration > stop_us:
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        timeline.append(
            {
                "name": str(event.get("name") or "unknown"),
                "start_us": timestamp - start_us,
                "duration_us": duration,
                "process": event.get("pid"),
                "thread": event.get("tid"),
                "device_id": 0,
                "grid": {
                    "x": int(_number(args.get("dimx"), 1)),
                    "y": int(_number(args.get("dimy"), 1)),
                    "z": int(_number(args.get("dimz"), 1)),
                },
                "kernel_type": args.get("kernel_type"),
                "visible_cluster": args.get("visible_cluster"),
                "local_dram_bytes": int(_number(args.get("ldram"))),
                "received_to_queued": args.get("received~queued"),
                "queued_to_submitted": args.get("queued~submitted"),
                "submitted_to_start": args.get("submitted~start"),
            }
        )
    timeline.sort(key=lambda item: item["start_us"])
    if not timeline:
        raise ValueError("CNPerf capture range did not contain device kernel events")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in timeline:
        grouped[event["name"]].append(event)
    kernels: List[Dict[str, Any]] = []
    for name, launches in grouped.items():
        durations = [item["duration_us"] for item in launches]
        kernels.append(
            {
                "name": name,
                "calls": len(launches),
                "total_duration_us": sum(durations),
                "average_duration_us": sum(durations) / len(durations),
                "minimum_duration_us": min(durations),
                "maximum_duration_us": max(durations),
                "grid_shapes": list(
                    {
                        json.dumps(item["grid"], sort_keys=True): item["grid"]
                        for item in launches
                    }.values()
                ),
                "kernel_types": sorted(
                    {str(item["kernel_type"]) for item in launches if item["kernel_type"]}
                ),
                "visible_clusters": sorted(
                    {
                        str(item["visible_cluster"])
                        for item in launches
                        if item["visible_cluster"]
                    }
                ),
            }
        )
    kernels.sort(key=lambda item: item["total_duration_us"], reverse=True)
    return {
        "schema_version": 1,
        "time_unit": "microseconds",
        "capture_range": {
            "source": "cnProfilerStart/Stop",
            "start_timestamp_us": start_us,
            "stop_timestamp_us": stop_us,
            "duration_us": stop_us - start_us,
        },
        "raw_event_count": len(events),
        "kernel_count": len(kernels),
        "kernel_invocation_count": len(timeline),
        "total_kernel_time_us": sum(item["duration_us"] for item in timeline),
        "kernels": kernels,
        "timeline": timeline,
    }


def parse_pmu_csv(path: Path) -> Dict[str, Any]:
    """Aggregate one CNPerf PMU CSV while preserving units and per-kernel values."""
    unit_name = path.stem
    if "_" in unit_name:
        prefix, suffix = unit_name.split("_", 1)
        if prefix.startswith("card"):
            unit_name = suffix

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field for field in (reader.fieldnames or []) if field]
        counters: Dict[str, tuple[str, str]] = {}
        for field in fields:
            if field in _COMMON_PMU_COLUMNS:
                continue
            match = _COUNTER_HEADER_RE.match(field)
            if match:
                counters[field] = (match.group("name"), match.group("unit"))
        rows = [row for row in reader if str(row.get("Kernel Name", "")).strip()]
    if not rows or not counters:
        return {}

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("Kernel Name", "")).strip().strip('"')].append(row)

    kernels: List[Dict[str, Any]] = []
    for kernel_name, kernel_rows in sorted(grouped.items()):
        normalized_counters = []
        for field, (name, value_unit) in counters.items():
            values = [_number(row.get(field)) for row in kernel_rows]
            normalized_counters.append(
                {
                    "name": name,
                    "unit": value_unit,
                    "sum": sum(values),
                    "average": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )
        kernels.append(
            {
                "name": kernel_name,
                "invocation_count": len(kernel_rows),
                "counters": normalized_counters,
            }
        )
    return {
        "unit": unit_name,
        "row_count": len(rows),
        "kernel_count": len(kernels),
        "counter_count": len(counters),
        "kernels": kernels,
    }


def parse_mlisa(
    text: str,
    *,
    binary_name: str,
) -> List[Dict[str, Any]]:
    """Parse compiler-emitted Cambricon MLISA without inventing PC addresses."""
    rows: List[Dict[str, Any]] = []
    kernel = ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        function = _MLISA_FUNCTION_RE.match(stripped)
        if function:
            kernel = function.group("name")
            continue
        if (
            not kernel
            or not stripped.endswith(";")
            or stripped.startswith((".", "//"))
        ):
            continue
        assembly = stripped[:-1].rstrip()
        if not assembly:
            continue
        rows.append(
            {
                "binary": binary_name,
                "kernel": kernel,
                "ordinal": len(rows),
                "mlisa_line": line_number,
                "opcode": assembly.split(None, 1)[0],
                "assembly": assembly,
            }
        )
    return rows


def _ir_opcode(line: str) -> str:
    expression = line.strip()
    if "=" in expression and expression.startswith("%"):
        expression = expression.split("=", 1)[1].lstrip()
    match = re.search(
        r"\b(?P<opcode>(?:tt|arith|math|scf|cf|memref|linalg(?:_ext)?|"
        r"mlu|mosa|llvm)\.[A-Za-z_][\w.]*)\b",
        expression,
    )
    return match.group("opcode") if match else ""


def parse_ir_source_operations(
    text: str,
    *,
    ir_name: str,
    target: ReportSourceTarget,
    kernel_hint: str = "",
) -> List[Dict[str, Any]]:
    """Build source-mapped compiler operations from TTIR/MLUIR locations."""
    locations: Dict[str, tuple[str, int, int]] = {}
    for line in text.splitlines():
        match = _IR_LOCATION_RE.match(line.strip())
        if match:
            locations[match.group("id")] = (
                match.group("path"),
                int(match.group("line")),
                int(match.group("column")),
            )

    submitted = {source.path for source in target.implementation.sources}
    rows: List[Dict[str, Any]] = []
    kernel = kernel_hint
    for ir_line, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        function = _IR_FUNCTION_RE.search(stripped)
        if function:
            kernel = function.group("name")
            continue
        if not kernel or stripped.startswith("#"):
            continue
        opcode = _ir_opcode(stripped)
        if not opcode:
            continue
        location: tuple[str, int, int] | None = None
        direct = _IR_DIRECT_LOCATION_RE.search(stripped)
        if direct:
            location = (
                direct.group("path"),
                int(direct.group("line")),
                int(direct.group("column")),
            )
        else:
            reference = _IR_LOCATION_REF_RE.search(stripped)
            if reference:
                location = locations.get(reference.group("id"))
        if location is None:
            continue
        source_file = _normalize_source_path(location[0], target)
        if source_file not in submitted:
            continue
        rows.append(
            {
                "ir_file": ir_name,
                "stage": Path(ir_name).suffix.lstrip("."),
                "kernel": kernel,
                "index": len(rows),
                "ir_line": ir_line,
                "opcode": opcode,
                "operation": stripped,
                "source_file": source_file,
                "source_line": location[1],
                "source_column": location[2],
            }
        )
    return rows


def _source_mapping_payload(
    operations: Iterable[Dict[str, Any]],
    target: ReportSourceTarget,
) -> Dict[str, Any]:
    source_lines = {
        source.path: source.content.splitlines() for source in target.implementation.sources
    }
    grouped: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for operation in operations:
        source_file = str(operation.get("source_file", ""))
        source_line = operation.get("source_line")
        if source_file not in source_lines or not isinstance(source_line, int):
            continue
        grouped[(source_file, source_line)].append(
            {
                "ir_file": operation["ir_file"],
                "stage": operation["stage"],
                "kernel": operation["kernel"],
                "index": operation["index"],
                "ir_line": operation["ir_line"],
                "opcode": operation["opcode"],
                "operation": operation["operation"],
                "source_column": operation["source_column"],
            }
        )
    locations = []
    for (source_file, source_line), rows in sorted(grouped.items()):
        content = source_lines[source_file]
        source_text = content[source_line - 1] if 0 < source_line <= len(content) else ""
        locations.append(
            {
                "source_file": source_file,
                "source_line": source_line,
                "source_text": source_text,
                "kernels": sorted({row["kernel"] for row in rows}),
                "operation_count": len(rows),
                "operations": rows,
            }
        )
    return {
        "evidence_type": "static_source_to_compiler_ir_mapping",
        "source_mapping_stage": "ttir",
        "assembly_stage": "mlisa",
        "mapping_granularity": "source-to-ttir-operation-and-kernel",
        "per_instruction_source_mapping": False,
        "source_location_count": len(locations),
        "mapped_compiler_operation_count": sum(
            item["operation_count"] for item in locations
        ),
        "source_locations": locations,
    }


def _render_instruction_summary(
    instructions: List[Dict[str, Any]],
    source_mapping: Dict[str, Any],
    *,
    mlisa_count: int,
    binary_count: int,
) -> str:
    counts = Counter(item["opcode"] for item in instructions)
    lines = [
        "Cambricon Triton-MLU Static Instruction Summary",
        "================================================",
        f"MLISA Files: {mlisa_count}",
        f"CNBIN Binaries: {binary_count}",
        f"MLISA Instructions: {len(instructions)}",
        f"Source-mapped TTIR Operations: "
        f"{source_mapping['mapped_compiler_operation_count']}",
        f"Source Locations: {source_mapping['source_location_count']}",
        "Native Instruction Assembly: available (MLISA)",
        "Post-link PC-addressed Disassembly: unavailable",
        "Per-instruction Source Mapping: unavailable",
        "Dynamic Instruction Timeline: unavailable",
        "",
        "Top MLISA Opcodes",
    ]
    lines.extend(f"  {opcode}: {count}" for opcode, count in counts.most_common(30))
    if not counts:
        lines.append("  No MLISA instructions were parsed.")
    return "\n".join(lines).rstrip() + "\n"
def _render_instruction_report(instructions: List[Dict[str, Any]]) -> str:
    lines = [
        "Cambricon Triton-MLU Instruction Listing",
        "==========================================",
        "MLISA is compiler-emitted native assembly, not post-link binary disassembly.",
        "Instruction ordinals below are not program-counter addresses.",
        "",
    ]
    current = None
    for instruction in instructions:
        key = (instruction["binary"], instruction["kernel"])
        if key != current:
            lines.extend([f"[{key[0]}] {key[1]}", ""])
            current = key
        lines.append(
            f"  {instruction['ordinal']:>5}  {instruction['opcode']:<28} "
            f"{instruction['assembly']}"
        )
    if not instructions:
        lines.append("No MLISA instructions were parsed.")
    return "\n".join(lines).rstrip() + "\n"

def _render_source_report(source_mapping: Dict[str, Any]) -> str:
    lines = [
        "Cambricon Triton Source Mapping",
        "================================",
        "Source locations map to TTIR operations and their containing MLISA kernel.",
        "They do not claim a one-to-one source-line to final instruction address mapping.",
        "",
    ]
    for location in source_mapping["source_locations"]:
        lines.append(
            f"{location['source_file']}:{location['source_line']} "
            f"({location['operation_count']} TTIR operations; "
            f"kernels={','.join(location['kernels'])})"
        )
        if location["source_text"]:
            lines.append(f"  {location['source_text']}")
        for operation in location["operations"]:
            lines.append(
                f"    {operation['opcode']:<24} {operation['operation']}"
            )
        lines.append("")
    if not source_mapping["source_locations"]:
        lines.append("No submitted-source TTIR mappings were found.")
    return "\n".join(lines).rstrip() + "\n"


def _counter_highlights(metrics: Dict[str, Any]) -> List[str]:
    wanted = {
        ("tp_cluster", "bandwidth_utils"),
        ("tp_core", "alu_cycles"),
        ("tp_core", "mv_inst_cycles"),
        ("tp_core", "simd_inst_executed"),
        ("tp_core", "simt_inst_executed"),
        ("tp_core", "inst_cache_miss"),
        ("tp_memcore", "dram_read_cycles"),
        ("tp_memcore", "dram_write_cycles"),
        ("llc", "hit_rate"),
    }
    highlights = []
    for unit in metrics.get("performance_counters", []):
        for kernel in unit.get("kernels", []):
            for counter in kernel.get("counters", []):
                if (unit.get("unit"), counter.get("name")) not in wanted:
                    continue
                highlights.append(
                    f"  {kernel['name']} {unit['unit']}.{counter['name']}: "
                    f"avg={counter['average']:.3f} {counter['unit']} "
                    f"sum={counter['sum']:.3f} {counter['unit']}"
                )
    return highlights[:30]


def _render_profile_details(
    target: ReportSourceTarget,
    options: ProfileOptions,
    device: str,
    physical_device: str,
    metrics: Dict[str, Any],
    instruction_summary: str = "",
) -> str:
    lines = [
        "==PROF== Cambricon CNPerf Profile",
        f"==PROF== Definition: {target.definition.name}",
        f"==PROF== Implementation: {target.implementation.name}",
        f"==PROF== Workload: {target.workload.name}",
        f"==PROF== Device: {device} (physical {physical_device})",
        "",
        "Section: Profile Overview",
        "",
        f"Definition: {target.definition.name}",
        f"Implementation: {target.implementation.name}",
        f"Workload: {target.workload.name}",
        f"Profile Level: {options.level}",
        f"Warmup Invocations: {options.warmup}",
        f"Measured Invocations: {options.iterations}",
        f"Logical Device: {device}",
        f"Physical Device: {physical_device}",
        "CNPerf Device after visibility binding: 0",
        "Capture Range: cnProfilerStart/Stop",
        "",
        "Section: Performance Signals",
        "",
        f"Kernel Rows: {metrics['kernel_count']}",
        f"Kernel Invocations: {metrics['kernel_invocation_count']}",
        f"Accumulated MLU Activity: {metrics['total_kernel_time_us']:.4f} us",
        f"PMU Units: {metrics.get('counter_unit_count', 0)}",
        f"PMU Counter Fields: {metrics.get('counter_metric_count', 0)}",
        "",
        "Top MLU Kernel Activities",
        "",
        "  Time(us)     Calls    Avg(us)    Min(us)    Max(us)  Name",
    ]
    for kernel in metrics["kernels"][:30]:
        lines.append(
            f"  {kernel['total_duration_us']:>9.3f}  {kernel['calls']:>8}  "
            f"{kernel['average_duration_us']:>9.3f}  "
            f"{kernel['minimum_duration_us']:>9.3f}  "
            f"{kernel['maximum_duration_us']:>9.3f}  {kernel['name']}"
        )
    lines.extend(["", "Selected PMU Counters", ""])
    highlights = _counter_highlights(metrics)
    lines.extend(highlights or ["  Not exported by this CNPerf run."])
    lines.extend(
        [
            "",
            "Section: Instruction Evidence",
            "",
            instruction_summary.rstrip()
            if instruction_summary
            else "Not requested at metrics level.",
            "",
            "Section: Interpretation",
            "",
            "CNPerf PMU collection is instrumented diagnostic evidence and is not eval latency.",
            "The normalized timeline retains only kernels inside cnProfilerStart/Stop.",
            "Non-replay PMU collection avoids re-executing stateful kernels but may expose only "
            "the counters collectable in one pass.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
