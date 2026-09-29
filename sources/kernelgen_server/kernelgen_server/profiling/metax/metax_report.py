# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""mcTracer and MACA compiler report normalization."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..models import ProfileOptions
from ..source_view import ReportSourceTarget


_TTIR_LOCATION_RE = re.compile(
    r'^#(?P<id>loc\d*)\s*=\s*loc\("(?P<path>[^"]+)":(?P<line>\d+):(?P<column>\d+)\)'
)
_TTIR_LOCATION_REF_RE = re.compile(r"loc\(#(?P<id>loc\d*)\)")
_TTIR_DIRECT_LOCATION_RE = re.compile(
    r'loc\("(?P<path>[^"]+)":(?P<line>\d+):(?P<column>\d+)\)'
)
_TTIR_FUNCTION_RE = re.compile(r"\btt\.func\b.*?@(?P<name>[A-Za-z_.$][\w.$-]*)")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_timestamp(event: Dict[str, Any]) -> int:
    try:
        return int(event.get("ts", 0))
    except (TypeError, ValueError):
        return 0


def _capture_window(events: Iterable[Dict[str, Any]]) -> tuple[int, int]:
    ordered = sorted(events, key=_event_timestamp)
    starts = [event for event in ordered if event.get("name") == "mcProfilerStart"]
    stops = [event for event in ordered if event.get("name") == "mcProfilerStop"]
    for start in starts:
        start_ns = _event_timestamp(start) + int(_number(start.get("dur")))
        stop = next(
            (event for event in stops if _event_timestamp(event) >= start_ns),
            None,
        )
        if stop is not None:
            return start_ns, _event_timestamp(stop)
    raise ValueError("mcTracer trace did not contain a complete mcProfilerStart/Stop range")


def _shape(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, int] = {}
    for axis in ("x", "y", "z"):
        try:
            result[axis] = int(value.get(axis, 1))
        except (TypeError, ValueError):
            result[axis] = 1
    return result


def _kernel_event(event: Dict[str, Any], start_ns: int, stop_ns: int) -> bool:
    args = event.get("args")
    if event.get("ph") != "X" or not isinstance(args, dict):
        return False
    if "device_id" not in args or "grid" not in args or "block" not in args:
        return False
    timestamp = _event_timestamp(event)
    duration = int(_number(event.get("dur")))
    return timestamp >= start_ns and timestamp + duration <= stop_ns


def parse_mctracer_trace(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize only device kernels inside the runtime profiler capture range."""
    raw_events = payload.get("traceEvents")
    if not isinstance(raw_events, list):
        raise ValueError("mcTracer JSON does not contain a traceEvents list")
    events = [event for event in raw_events if isinstance(event, dict)]
    start_ns, stop_ns = _capture_window(events)

    timeline: List[Dict[str, Any]] = []
    for event in events:
        if not _kernel_event(event, start_ns, stop_ns):
            continue
        args = event["args"]
        memory = args.get("mem") if isinstance(args.get("mem"), dict) else {}
        timestamp = _event_timestamp(event)
        duration_ns = int(_number(event.get("dur")))
        timeline.append(
            {
                "name": str(event.get("name") or args.get("name") or "unknown"),
                "start_us": (timestamp - start_ns) / 1_000.0,
                "duration_us": duration_ns / 1_000.0,
                "device_id": args.get("device_id"),
                "queue_id": args.get("queue_id"),
                "hardware_queue_ids": args.get("hw_queue_id", []),
                "grid": _shape(args.get("grid")),
                "block": _shape(args.get("block")),
                "registers_per_thread": int(_number(memory.get("registers_per_thread"))),
                "static_shared_bytes": int(_number(memory.get("static_shared"))),
                "dynamic_shared_bytes": int(_number(memory.get("dynamic_shared"))),
                "private_bytes_per_thread": int(_number(memory.get("private_per_thread"))),
                "private_bytes_total": int(_number(memory.get("private_total"))),
                "register_occupancy_percent": _number(args.get("mtreg_occupancy(%)")),
                "shared_memory_occupancy_percent": _number(
                    args.get("shared_memeory_occupancy(%)")
                ),
                "dispatch_id": args.get("dispatch_id"),
                "correlation_id": args.get("co_id"),
            }
        )
    timeline.sort(key=lambda item: item["start_us"])
    if not timeline:
        raise ValueError("mcTracer capture range did not contain any device kernel events")

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
                "maximum_registers_per_thread": max(
                    item["registers_per_thread"] for item in launches
                ),
                "maximum_static_shared_bytes": max(
                    item["static_shared_bytes"] for item in launches
                ),
                "maximum_dynamic_shared_bytes": max(
                    item["dynamic_shared_bytes"] for item in launches
                ),
                "grid_shapes": list(
                    {
                        json.dumps(item["grid"], sort_keys=True): item["grid"]
                        for item in launches
                    }.values()
                ),
                "block_shapes": list(
                    {
                        json.dumps(item["block"], sort_keys=True): item["block"]
                        for item in launches
                    }.values()
                ),
            }
        )
    kernels.sort(key=lambda item: item["total_duration_us"], reverse=True)
    return {
        "schema_version": 1,
        "time_unit": "microseconds",
        "capture_range": {
            "source": "mcProfilerStart/Stop",
            "start_timestamp_ns": int(start_ns),
            "stop_timestamp_ns": int(stop_ns),
            "duration_us": (stop_ns - start_ns) / 1_000.0,
        },
        "raw_event_count": len(events),
        "kernel_count": len(kernels),
        "kernel_invocation_count": len(timeline),
        "total_kernel_time_us": sum(item["duration_us"] for item in timeline),
        "kernels": kernels,
        "timeline": timeline,
    }


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


def _ttir_opcode(line: str) -> str:
    expression = line.strip()
    if "=" in expression and expression.startswith("%"):
        expression = expression.split("=", 1)[1].lstrip()
    match = re.match(r"(?P<opcode>[A-Za-z_][\w.]*)\b", expression)
    return match.group("opcode") if match else ""


def parse_ttir_instructions(
    text: str,
    *,
    binary_name: str,
    target: ReportSourceTarget,
) -> List[Dict[str, Any]]:
    """Build a source-mapped compiler-operation listing from Triton TTIR."""
    locations: Dict[str, tuple[str, int, int]] = {}
    for line in text.splitlines():
        match = _TTIR_LOCATION_RE.match(line.strip())
        if match:
            locations[match.group("id")] = (
                match.group("path"),
                int(match.group("line")),
                int(match.group("column")),
            )

    rows: List[Dict[str, Any]] = []
    kernel = ""
    for line in text.splitlines():
        stripped = line.strip()
        function_match = _TTIR_FUNCTION_RE.search(stripped)
        if function_match:
            kernel = function_match.group("name")
            continue
        if not kernel or stripped.startswith("#") or stripped in {"}", "} loc(#loc)"}:
            continue
        opcode = _ttir_opcode(stripped)
        if not opcode or opcode in {"module", "attributes"}:
            continue
        location: tuple[str, int, int] | None = None
        direct = _TTIR_DIRECT_LOCATION_RE.search(stripped)
        if direct:
            location = (
                direct.group("path"),
                int(direct.group("line")),
                int(direct.group("column")),
            )
        else:
            reference = _TTIR_LOCATION_REF_RE.search(stripped)
            if reference:
                location = locations.get(reference.group("id"))
        source_file = _normalize_source_path(location[0], target) if location else ""
        rows.append(
            {
                "binary": binary_name,
                "kernel": kernel,
                "stage": "ttir",
                "index": len(rows),
                "opcode": opcode,
                "instruction": stripped,
                "source_file": source_file,
                "source_line": location[1] if location else None,
                "source_column": location[2] if location else None,
            }
        )
    return rows


def _source_mapping_payload(
    instructions: Iterable[Dict[str, Any]],
    target: ReportSourceTarget,
) -> Dict[str, Any]:
    source_lines = {
        source.path: source.content.splitlines() for source in target.implementation.sources
    }
    grouped: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for instruction in instructions:
        source_file = str(instruction.get("source_file", ""))
        source_line = instruction.get("source_line")
        if source_file not in source_lines or not isinstance(source_line, int):
            continue
        grouped[(source_file, source_line)].append(
            {
                "binary": instruction["binary"],
                "kernel": instruction["kernel"],
                "stage": instruction["stage"],
                "index": instruction["index"],
                "opcode": instruction["opcode"],
                "instruction": instruction["instruction"],
                "source_column": instruction.get("source_column"),
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
                "instruction_count": len(rows),
                "instructions": rows,
            }
        )
    return {
        "evidence_type": "static_compiler_source_mapping",
        "instruction_stage": "ttir",
        "source_location_count": len(locations),
        "mapped_instruction_count": sum(item["instruction_count"] for item in locations),
        "source_locations": locations,
    }


def _render_instruction_summary(
    instructions: List[Dict[str, Any]],
    source_mapping: Dict[str, Any],
    *,
    binary_count: int,
    object_count: int,
) -> str:
    counts = Counter(item["opcode"] for item in instructions)
    lines = [
        "MetaX Static Compiler Instruction Summary",
        "=========================================",
        f"mcfatbin Binaries: {binary_count}",
        f"Device Objects Extracted: {object_count}",
        f"TTIR Instructions: {len(instructions)}",
        f"Source-mapped Instructions: {source_mapping['mapped_instruction_count']}",
        f"Source Locations: {source_mapping['source_location_count']}",
        "Native Machine ISA Listing: unavailable in the installed MACA SDK",
        "Dynamic Instruction Timeline: unavailable",
        "",
        "Top TTIR Opcodes",
    ]
    lines.extend(f"  {opcode}: {count}" for opcode, count in counts.most_common(30))
    if not counts:
        lines.append("  No TTIR instructions were parsed.")
    return "\n".join(lines).rstrip() + "\n"
def _render_instruction_report(instructions: List[Dict[str, Any]]) -> str:
    lines = [
        "MetaX Triton Compiler Instruction Listing",
        "==========================================",
        "This is TTIR compiler evidence, not native MetaX machine ISA.",
        "",
    ]
    current = None
    for instruction in instructions:
        key = (instruction["binary"], instruction["kernel"])
        if key != current:
            lines.extend([f"[{key[0]}] {key[1]}", ""])
            current = key
        location = ""
        if instruction.get("source_file") and instruction.get("source_line") is not None:
            location = f" {instruction['source_file']}:{instruction['source_line']}"
        lines.append(
            f"  {instruction['index']:>5}  {instruction['opcode']:<24}"
            f"{location}\n         {instruction['instruction']}"
        )
    if not instructions:
        lines.append("No TTIR instructions were parsed.")
    return "\n".join(lines).rstrip() + "\n"

def _render_source_report(source_mapping: Dict[str, Any]) -> str:
    lines = ["MetaX Triton Source Mapping", "============================", ""]
    for location in source_mapping["source_locations"]:
        lines.append(
            f"{location['source_file']}:{location['source_line']} "
            f"({location['instruction_count']} TTIR instructions)"
        )
        if location["source_text"]:
            lines.append(f"  {location['source_text']}")
        for instruction in location["instructions"]:
            lines.append(
                f"    {instruction['index']:>5}  {instruction['opcode']:<24}"
                f"{instruction['instruction']}"
            )
        lines.append("")
    if not source_mapping["source_locations"]:
        lines.append("No submitted-source mappings were found.")
    return "\n".join(lines).rstrip() + "\n"


def _hex_dump(data: bytes) -> str:
    lines = []
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        lines.append(f"{offset:08x}  {' '.join(f'{byte:02x}' for byte in chunk)}")
    return "\n".join(lines).rstrip() + "\n"


def _render_profile_details(
    target: ReportSourceTarget,
    options: ProfileOptions,
    device: str,
    physical_device: str,
    metrics: Dict[str, Any],
    instruction_summary: str = "",
) -> str:
    lines = [
        "==PROF== MetaX mcTracer Profile",
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
        "Capture Range: mcProfilerStart/Stop",
        "",
        "Section: Performance Signals",
        "",
        f"Kernel Rows: {metrics['kernel_count']}",
        f"Kernel Invocations: {metrics['kernel_invocation_count']}",
        f"Accumulated Device Activity: {metrics['total_kernel_time_us']:.4f} us",
        "",
        "Top MetaX Kernel Activities",
        "",
        "  Time(us)     Calls    Avg(us)  Registers  StaticShmem  Name",
    ]
    for kernel in metrics["kernels"][:30]:
        lines.append(
            f"  {kernel['total_duration_us']:>9.3f}  {kernel['calls']:>8}  "
            f"{kernel['average_duration_us']:>9.3f}  "
            f"{kernel['maximum_registers_per_thread']:>9}  "
            f"{kernel['maximum_static_shared_bytes']:>11}  {kernel['name']}"
        )
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
            "mcTracer activity timing is instrumented diagnostics and is not eval latency.",
            "The normalized timeline excludes warmup using mcProfilerStart/Stop markers.",
            "The installed mcTracer mctx export does not include hardware counter sets.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
