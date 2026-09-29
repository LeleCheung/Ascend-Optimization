# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""topsprof and GCU instruction report normalization."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..models import ProfileOptions
from ..source_view import ReportSourceTarget


_SOURCE_LOCATION_RE = re.compile(r"^;\s+(.+?):(\d+)(?::\d+)?\s*$")
_FUNCTION_RE = re.compile(r"^[0-9a-fA-F]+\s+<(.+)>:$")
_INSTRUCTION_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+)+)\s*(\S.*)?$"
)
_TRACE_ROW_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(.+?)\s*$")
_BUNDLE_ARCH_RE = re.compile(r"--(gcu\d+)$")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default


def parse_topsprof_summary(path: Path) -> Dict[str, Any]:
    """Normalize the stable topsprof summary CSV emitted by VPD 3.x."""
    kernels: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Type", "Time(%)", "Time(us)", "Calls", "Avg(us)", "Name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "topsprof CSV did not contain the expected summary columns: "
                + ", ".join(sorted(required))
            )
        for row in reader:
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            calls = max(0, int(_number(row.get("Calls"))))
            kernels.append(
                {
                    "name": name,
                    "activity_type": str(row.get("Type", "")).strip(),
                    "time_percent": _number(row.get("Time(%)")),
                    "total_duration_us": _number(row.get("Time(us)")),
                    "calls": calls,
                    "average_duration_us": _number(row.get("Avg(us)")),
                    "minimum_duration_us": _number(row.get("Min(us)")),
                    "maximum_duration_us": _number(row.get("Max(us)")),
                }
            )
    if not kernels:
        raise ValueError("topsprof CSV did not contain any GCU activities")
    kernels.sort(key=lambda item: item["total_duration_us"], reverse=True)
    return {
        "time_unit": "microseconds",
        "kernel_count": len(kernels),
        "kernel_invocation_count": sum(item["calls"] for item in kernels),
        "total_kernel_time_us": sum(item["total_duration_us"] for item in kernels),
        "kernels": kernels,
    }


def parse_topsprof_trace(text: str) -> List[Dict[str, Any]]:
    """Parse ``topsprof --print-gcu-trace`` text when CSV trace export is empty."""
    rows: List[Dict[str, Any]] = []
    active = False
    for line in text.splitlines():
        if line.strip() == "GCU trace:":
            active = True
            continue
        if not active:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("Start("):
            continue
        if stripped.startswith("Accumulate Time") or stripped.startswith("Total Time"):
            continue
        match = _TRACE_ROW_RE.match(line)
        if match:
            rows.append(
                {
                    "start_us": float(match.group(1)),
                    "duration_us": float(match.group(2)),
                    "name": match.group(3),
                }
            )
    return rows


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


def parse_instruction_dump(
    text: str,
    *,
    binary_name: str,
    target: ReportSourceTarget,
) -> List[Dict[str, Any]]:
    """Parse LLVM's GCU disassembly while retaining source-to-PC evidence."""
    rows: List[Dict[str, Any]] = []
    function = ""
    source_file = ""
    source_line: int | None = None
    submitted_paths = {source.path for source in target.implementation.sources}
    for line in text.splitlines():
        function_match = _FUNCTION_RE.match(line.strip())
        if function_match:
            function = function_match.group(1)
            source_file = ""
            source_line = None
            continue
        source_match = _SOURCE_LOCATION_RE.match(line.strip())
        if source_match:
            source_file = _normalize_source_path(source_match.group(1), target)
            source_line = int(source_match.group(2))
            continue
        instruction_match = _INSTRUCTION_RE.match(line)
        if not instruction_match or not function:
            continue
        assembly = (instruction_match.group(3) or "").strip()
        if not assembly or assembly == "...":
            continue
        rows.append(
            {
                "binary": binary_name,
                "kernel": function,
                "pc": f"0x{instruction_match.group(1).lower()}",
                "opcode": assembly.split(None, 1)[0],
                "assembly": assembly,
                "source_file": source_file,
                "source_line": source_line,
                "submitted_source": source_file in submitted_paths,
            }
        )

    stem = Path(binary_name).stem
    sourced_functions = {
        row["kernel"] for row in rows if row["submitted_source"]
    }
    named_functions = {
        row["kernel"]
        for row in rows
        if stem in row["kernel"] or row["kernel"] in stem
    }
    selected = sourced_functions or named_functions
    if not selected:
        selected = {row["kernel"] for row in rows if not row["kernel"].startswith("__")}
    for row in rows:
        row.pop("submitted_source", None)
    return [row for row in rows if row["kernel"] in selected]


def _source_mapping_payload(
    instructions: Iterable[Dict[str, Any]],
    target: ReportSourceTarget,
) -> Dict[str, Any]:
    source_lines = {
        source.path: source.content.splitlines()
        for source in target.implementation.sources
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
                "pc": instruction["pc"],
                "opcode": instruction["opcode"],
                "assembly": instruction["assembly"],
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
        "evidence_type": "static_source_mapping",
        "source_location_count": len(locations),
        "mapped_instruction_count": sum(item["instruction_count"] for item in locations),
        "source_locations": locations,
    }


def _render_instruction_summary(
    instructions: List[Dict[str, Any]],
    source_mapping: Dict[str, Any],
    *,
    binary_count: int,
) -> str:
    counts = Counter(item["opcode"] for item in instructions)
    lines = [
        "Enflame GCU Static Instruction Summary",
        "=======================================",
        f"Binaries Disassembled: {binary_count}",
        f"Instructions: {len(instructions)}",
        f"Source-mapped Instructions: {source_mapping['mapped_instruction_count']}",
        f"Source Locations: {source_mapping['source_location_count']}",
        "Dynamic Instruction Timeline: unavailable",
        "",
        "Top Opcodes",
    ]
    lines.extend(f"  {opcode}: {count}" for opcode, count in counts.most_common(30))
    if not counts:
        lines.append("  No instructions were parsed.")
    return "\n".join(lines).rstrip() + "\n"
def _render_instruction_report(instructions: List[Dict[str, Any]]) -> str:
    lines = [
        "Enflame GCU Instruction Listing",
        "================================",
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
            f"  {instruction['pc']:>12}  {instruction['assembly']}{location}"
        )
    return "\n".join(lines).rstrip() + "\n"

def _render_source_report(source_mapping: Dict[str, Any]) -> str:
    lines = ["Enflame GCU Source Mapping", "===========================", ""]
    for location in source_mapping["source_locations"]:
        lines.append(
            f"{location['source_file']}:{location['source_line']} "
            f"({location['instruction_count']} instructions)"
        )
        if location["source_text"]:
            lines.append(f"  {location['source_text']}")
        for instruction in location["instructions"]:
            lines.append(
                f"    {instruction['pc']:>12}  {instruction['assembly']}"
            )
        lines.append("")
    if not source_mapping["source_locations"]:
        lines.append("No submitted-source mappings were found.")
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
        "==PROF== Enflame topsprof Profile",
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
        "",
        "Section: Performance Signals",
        "",
        f"Kernel Rows: {metrics['kernel_count']}",
        f"Kernel Invocations: {metrics['kernel_invocation_count']}",
        f"Accumulated GCU Activity: {metrics['total_kernel_time_us']:.4f} us",
        "",
        "Top GCU Activities",
        "",
        "  Time(us)     Calls    Avg(us)  Name",
    ]
    for kernel in metrics["kernels"][:30]:
        lines.append(
            f"  {kernel['total_duration_us']:>9.3f}  {kernel['calls']:>8}  "
            f"{kernel['average_duration_us']:>9.3f}  {kernel['name']}"
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
            "topsprof durations are instrumented diagnostics and are not eval latency.",
            "This topsprof build exports activity timing, not hardware counter sets.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
