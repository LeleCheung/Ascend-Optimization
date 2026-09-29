# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Parsing and factual rendering for Hygon hipprof/rocprof reports."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..models import ProfileOptions
from ..source_view import ReportSourceTarget


_PMC_METADATA_COLUMNS = {
    "kernel-name",
    "dispatch",
    "gpu-id",
    "queue-id",
    "queue-index",
    "tid",
    "grd",
    "wgr",
    "lds",
    "scr",
    "vgpr",
    "sgpr",
    "fbar",
    "sig",
    "time",
}
_NON_KERNEL_NAMES = {
    "memcpy",
    "memset",
    "hosttodevice",
    "devicetohost",
    "devicetodevice",
    "copy",
}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [dict(row) for row in rows if row]


def _is_kernel_name(name: str) -> bool:
    compact = "".join(name.lower().split())
    return bool(name.strip()) and compact not in _NON_KERNEL_NAMES


def _build_operations(
    samples_by_name: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    warmup: int,
    source: str,
) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    total_avg = 0.0
    expected = warmup + iterations
    exact_names = {
        name for name, samples in samples_by_name.items() if len(samples) == expected
    }
    if exact_names:
        selected_names = exact_names
    else:
        selected_names = {
            name for name, samples in samples_by_name.items() if len(samples) >= iterations
        }

    if not selected_names:
        raise ValueError("no profiled kernels have enough invocations for the requested iterations")

    for name, raw_samples in samples_by_name.items():
        if name not in selected_names:
            continue
        samples = list(raw_samples)
        samples = samples[-iterations:]
        if not samples or any(value < 0 for value in samples):
            continue
        avg_duration = sum(samples) / len(samples)
        total_avg += avg_duration
        operations.append(
            {
                "op_name": name,
                "measured_invocations": len(samples),
                "total_duration_us": round(sum(samples), 3),
                "avg_duration_us": round(avg_duration, 3),
                "min_duration_us": round(min(samples), 3),
                "max_duration_us": round(max(samples), 3),
                "timing_source": source,
            }
        )

    if not operations:
        raise ValueError("no valid GPU kernel duration samples were found")
    return {
        "avg_time_us": round(total_avg, 3),
        "ops": operations,
        "timing_source": source,
    }


def parse_hipprof_timeline(path: Path, warmup: int, iterations: int) -> dict[str, Any]:
    """Parse per-dispatch GPU kernel events and exclude warmup invocations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("timeline does not contain a traceEvents list")

    samples: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        args = event.get("args")
        if not isinstance(args, dict) or not ({"dev_id", "dev-id"} & args.keys()):
            continue
        name = str(event.get("name") or args.get("Name") or "").strip()
        if not _is_kernel_name(name):
            continue
        duration_ns = _as_float(args.get("DurationNs"))
        if duration_ns is None:
            begin_ns = _as_float(args.get("BeginNs"))
            end_ns = _as_float(args.get("EndNs"))
            if begin_ns is not None and end_ns is not None:
                duration_ns = end_ns - begin_ns
        if duration_ns is not None and duration_ns >= 0:
            samples[name].append(duration_ns / 1_000.0)

    return _build_operations(
        samples,
        iterations=iterations,
        warmup=warmup,
        source="timeline",
    )


def parse_hipprof_kernel_csv(path: Path, warmup: int, iterations: int) -> dict[str, Any]:
    """Parse hipprof's aggregate kernel CSV as a timeline fallback."""
    rows = _read_csv(path)
    if not rows:
        raise ValueError("kernel CSV is empty")
    operations: list[dict[str, Any]] = []
    total_avg = 0.0
    for row in rows:
        name = str(row.get("Name") or row.get("KernelName") or row.get("kernel-name") or "")
        if name.strip().lower() == "total" or not _is_kernel_name(name):
            continue
        calls_value = _as_float(row.get("Calls"))
        average_ns = _as_float(row.get("AverageNs"))
        duration_ns = _as_float(row.get("DurationNs"))
        if calls_value is not None and average_ns is not None:
            calls = int(calls_value)
            if calls < iterations:
                continue
            average_us = average_ns / 1_000.0
            operations.append(
                {
                    "op_name": name,
                    "measured_invocations": iterations,
                    "total_duration_us": round(average_us * iterations, 3),
                    "avg_duration_us": round(average_us, 3),
                    "min_duration_us": round(average_us, 3),
                    "max_duration_us": round(average_us, 3),
                    "timing_source": "kernel_csv",
                }
            )
            total_avg += average_us
        elif duration_ns is not None:
            duration_us = duration_ns / 1_000.0
            operations.append(
                {
                    "op_name": name,
                    "measured_invocations": 1,
                    "total_duration_us": round(duration_us, 3),
                    "avg_duration_us": round(duration_us, 3),
                    "min_duration_us": round(duration_us, 3),
                    "max_duration_us": round(duration_us, 3),
                    "timing_source": "kernel_csv",
                }
            )
            total_avg += duration_us
    if not operations:
        raise ValueError("kernel CSV has no valid timed kernel rows")
    return {
        "avg_time_us": round(total_avg, 3),
        "ops": operations,
        "timing_source": "kernel_csv",
    }


def parse_hipprof_pmc_csv(path: Path) -> dict[str, Any]:
    """Parse at least one valid hipprof PMC row; ``time`` is in seconds."""
    rows = _read_csv(path)
    if not rows:
        raise ValueError("PMC CSV is empty")
    parsed_rows: list[dict[str, Any]] = []
    counter_names: list[str] = []
    for row in rows:
        name = str(row.get("kernel-name") or "").strip()
        time_s = _as_float(row.get("time"))
        if not _is_kernel_name(name) or time_s is None or time_s < 0:
            continue
        counters: dict[str, float] = {}
        for key, raw_value in row.items():
            if key in _PMC_METADATA_COLUMNS:
                continue
            value = _as_float(raw_value)
            if value is not None:
                counters[key] = value
                if key not in counter_names:
                    counter_names.append(key)
        parsed_rows.append(
            {
                "kernel_name": name,
                "dispatch": int(_as_float(row.get("dispatch")) or 0),
                "gpu_id": str(row.get("gpu-id") or ""),
                "duration_us": round(time_s * 1_000_000.0, 3),
                "grid_size": int(_as_float(row.get("grd")) or 0),
                "workgroup_size": int(_as_float(row.get("wgr")) or 0),
                "lds_bytes": int(_as_float(row.get("lds")) or 0),
                "vgpr": int(_as_float(row.get("vgpr")) or 0),
                "sgpr": int(_as_float(row.get("sgpr")) or 0),
                "counters": counters,
            }
        )
    if not parsed_rows:
        raise ValueError("PMC CSV has no valid kernel records")
    return {
        "records": parsed_rows,
        "record_count": len(parsed_rows),
        "counter_names": counter_names,
    }


def select_measured_hipprof_pmc_records(
    pmc: Mapping[str, Any], parsed: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Keep PMC rows for the measured target-kernel invocations only."""
    remaining = {
        str(operation.get("op_name", "")): int(
            operation.get("measured_invocations", 0)
        )
        for operation in parsed.get("ops", [])
        if str(operation.get("op_name", ""))
        and int(operation.get("measured_invocations", 0)) > 0
    }
    selected: list[dict[str, Any]] = []
    for record in reversed(list(pmc.get("records", []))):
        kernel_name = str(record.get("demangled_name") or record.get("kernel_name", ""))
        if remaining.get(kernel_name, 0) <= 0:
            continue
        selected.append(dict(record))
        remaining[kernel_name] -= 1
    selected.reverse()
    if not selected:
        return None

    counter_names: list[str] = []
    for record in selected:
        for name in record.get("counters", {}):
            if name not in counter_names:
                counter_names.append(name)
    return {
        "records": selected,
        "record_count": len(selected),
        "raw_record_count": int(pmc.get("record_count", len(selected))),
        "counter_names": counter_names,
    }


def parse_rocprof_csv(path: Path, warmup: int, iterations: int) -> dict[str, Any]:
    """Parse rocprof dispatch CSV when it includes valid duration columns."""
    rows = _read_csv(path)
    if not rows:
        raise ValueError("rocprof CSV is empty")
    samples: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        name = str(row.get("KernelName") or row.get("Name") or "").strip()
        if not _is_kernel_name(name):
            continue
        duration_ns = _as_float(row.get("DurationNs"))
        if duration_ns is None:
            begin_ns = _as_float(row.get("BeginNs"))
            end_ns = _as_float(row.get("EndNs"))
            if begin_ns is not None and end_ns is not None:
                duration_ns = end_ns - begin_ns
        if duration_ns is not None and duration_ns >= 0:
            samples[name].append(duration_ns / 1_000.0)
    return _build_operations(
        samples,
        iterations=iterations,
        warmup=warmup,
        source="rocprof_csv",
    )


def validate_timeline(path: Path) -> int:
    """Return the number of complete device events or reject an empty trace."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("timeline does not contain a traceEvents list")
    count = sum(
        1
        for event in events
        if isinstance(event, dict)
        and event.get("ph") == "X"
        and isinstance(event.get("args"), dict)
        and bool({"dev_id", "dev-id"} & event["args"].keys())
    )
    if count == 0:
        raise ValueError("timeline contains no complete device events")
    return count


def _clean(value: Any) -> str:
    return str(value if value is not None else "").replace("\n", " ").strip()


def _render_table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    normalized = [[_clean(value) for value in row] for row in rows]
    widths = [len(column) for column in columns]
    for row in normalized:
        for index, value in enumerate(row[: len(widths)]):
            widths[index] = max(widths[index], len(value))

    def line(row: Sequence[str]) -> str:
        values = list(row) + [""] * (len(widths) - len(row))
        return "    " + "  ".join(values[index].ljust(widths[index]) for index in range(len(widths))).rstrip()

    output = [line(columns), "    " + "  ".join("-" * width for width in widths)]
    output.extend(line(row) for row in normalized)
    return output


def render_hygon_details(
    target: ReportSourceTarget,
    options: ProfileOptions,
    device: str,
    profiler_name: str,
    parsed: Mapping[str, Any],
    pmc: Mapping[str, Any] | None,
    pmc_requested: bool,
) -> str:
    """Render a factual report shaped like the other KernelGen profilers."""
    lines = [
        f"==PROF== Hygon DCU {profiler_name} Profile",
        f"==PROF== Definition: {target.definition.name}",
        f"==PROF== Implementation: {target.implementation.name}",
        f"==PROF== Workload: {target.workload.name}",
        f"==PROF== Device: {device}",
        "",
        "Section: Overview",
        "",
    ]
    lines.extend(
        _render_table(
            ["Field", "Value"],
            [
                ("Profiler", profiler_name),
                ("Profile Level", options.level),
                ("Warmup Invocations", options.warmup),
                ("Measured Invocations", options.iterations),
                ("Kernel Invocations", parsed.get("kernel_invocation_count", 0)),
                ("Distinct Kernel Operations", parsed.get("operation_count", 0)),
                ("Timing Source", parsed.get("timing_source", "")),
                ("Average Time (us)", parsed.get("avg_time_us", "")),
                ("Timing Authority", "diagnostic only; evaluator latency is authoritative"),
            ],
        )
    )
    lines.extend(["", "Section: Kernel Summary", ""])
    lines.extend(
        _render_table(
            ["Kernel", "Calls", "Average (us)", "Minimum (us)", "Maximum (us)"],
            (
                (
                    operation.get("op_name", ""),
                    operation.get("measured_invocations", ""),
                    operation.get("avg_duration_us", ""),
                    operation.get("min_duration_us", ""),
                    operation.get("max_duration_us", ""),
                )
                for operation in parsed.get("ops", [])
            ),
        )
    )
    lines.extend(["", "Section: Performance Signals", ""])
    if pmc and pmc.get("records"):
        lines.append(f"    Record Count: {pmc.get('record_count', 0)}")
        lines.append(f"    Counter Names: {', '.join(pmc.get('counter_names', []))}")
        lines.append("")
        for index, record in enumerate(pmc["records"], start=1):
            lines.append(f"    [{index}] {record.get('kernel_name', '')}")
            rows = [("duration_us", record.get("duration_us", ""))]
            rows.extend(sorted(record.get("counters", {}).items()))
            lines.extend(_render_table(["Metric", "Value"], rows))
            lines.append("")
    elif pmc_requested:
        lines.extend(
            [
                "    PMC collection was requested, but the profiler exported no valid records.",
                "",
            ]
        )
    else:
        lines.extend(
            ["    PMC collection was not requested for this profiler and level.", ""]
        )
    return "\n".join(lines).rstrip() + "\n"


_HYGON_FUNCTION = re.compile(r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:\s*$")
_HYGON_SOURCE = re.compile(r"^\s*;\s+(.+?):(\d+)(?::\d+)?\s*$")
_HYGON_INSTRUCTION = re.compile(
    r"^\s*(.*?)\s+//\s*([0-9a-fA-F]+):\s*([0-9a-fA-F ]+)\s*$"
)


def _normalize_source_path(path: str, target: ReportSourceTarget) -> str:
    reported = Path(path).as_posix()
    exact = {
        Path(source.path).as_posix(): source.path
        for source in target.implementation.sources
    }
    if reported in exact:
        return exact[reported]
    matches = [
        source.path
        for source in target.implementation.sources
        if reported.endswith("/" + Path(source.path).as_posix())
        or Path(reported).name == Path(source.path).name
    ]
    return matches[0] if len(set(matches)) == 1 else reported


def parse_hygon_isa(
    text: str, *, binary: Path, target: ReportSourceTarget
) -> list[dict[str, Any]]:
    """Parse DTK llvm-objdump static ISA with optional source line comments."""
    source_lines = {
        source.path: source.content.splitlines()
        for source in target.implementation.sources
    }
    function = ""
    source_file = ""
    source_line: int | None = None
    instructions: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        function_match = _HYGON_FUNCTION.match(raw_line)
        if function_match:
            function = function_match.group(2)
            source_file = ""
            source_line = None
            continue
        source_match = _HYGON_SOURCE.match(raw_line)
        if source_match:
            source_file = _normalize_source_path(source_match.group(1), target)
            source_line = int(source_match.group(2))
            continue
        instruction_match = _HYGON_INSTRUCTION.match(raw_line)
        if not instruction_match or not function:
            continue
        instruction_text = instruction_match.group(1).strip()
        opcode, _, operands = instruction_text.partition(" ")
        source_text = ""
        if source_file in source_lines and source_line is not None:
            lines = source_lines[source_file]
            if 1 <= source_line <= len(lines):
                source_text = lines[source_line - 1]
        instructions.append(
            {
                "binary": binary.name,
                "function_name": function,
                "pc": "0x" + instruction_match.group(2).lower(),
                "opcode": opcode,
                "operands": operands.strip(),
                "text": instruction_text,
                "encoding": " ".join(instruction_match.group(3).split()),
                "source_file": source_file,
                "source_line": source_line,
                "source_text": source_text,
            }
        )
    return instructions


def build_hygon_source_mapping(
    instructions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    locations: dict[tuple[str, int], dict[str, Any]] = {}
    for instruction in instructions:
        source_file = str(instruction.get("source_file", ""))
        source_line = instruction.get("source_line")
        if not source_file or source_line is None:
            continue
        key = (source_file, int(source_line))
        location = locations.setdefault(
            key,
            {
                "file": source_file,
                "line": int(source_line),
                "source_text": instruction.get("source_text", ""),
                "instructions": [],
            },
        )
        location["instructions"].append(
            {
                key: instruction.get(key, "")
                for key in ("binary", "function_name", "pc", "opcode", "operands")
            }
        )
    source_locations = sorted(
        locations.values(), key=lambda item: (item["file"], item["line"])
    )
    return {
        "schema_version": 1,
        "profiler": "dtk-llvm-objdump",
        "evidence_type": "static_source_mapping",
        "source_location_count": len(source_locations),
        "mapped_instruction_count": sum(
            len(location["instructions"]) for location in source_locations
        ),
        "source_locations": source_locations,
    }


def render_hygon_instruction_summary(
    instructions: Sequence[Mapping[str, Any]],
    *,
    binary_count: int,
    disassembled_binary_count: int,
) -> str:
    mapped_count = sum(
        bool(instruction.get("source_file"))
        and instruction.get("source_line") is not None
        for instruction in instructions
    )
    opcode_counts = Counter(str(item.get("opcode", "")) for item in instructions)
    lines = [
        "Hygon DTK Static Instruction Summary",
        "====================================",
        f"Binaries Found: {binary_count}",
        f"Binaries Disassembled: {disassembled_binary_count}",
        f"Instructions: {len(instructions)}",
        f"Source-mapped Instructions: {mapped_count}",
        "Dynamic Instruction Timeline: unavailable",
        "",
        "Top Opcodes",
    ]
    for opcode, count in opcode_counts.most_common(20):
        lines.append(f"  {opcode}: {count}")
    if not opcode_counts:
        lines.append("  No instructions were parsed.")
    return "\n".join(lines).rstrip() + "\n"
