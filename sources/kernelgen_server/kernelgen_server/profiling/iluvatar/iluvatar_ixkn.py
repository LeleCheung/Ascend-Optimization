# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Structured source and counter reports exported from IXKN profiles."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

IXKN_SECTION_ORDER = (
    "GPU Speed Of Light",
    "Compute Workload Analysis",
    "Memory Access",
    "Scheduler Statistics",
    "Warp State Statistics",
    "Instruction Statistics",
    "Launch Statistics",
    "Occupancy",
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_FILE_PATH = re.compile(r"^\s*File Path:\s*(.*?)\s*$")
_FUNCTION_NAME = re.compile(r"^\s*Function Name:\s*(.*?)\s*$")
_KERNEL_NAME = re.compile(r"^\s*Kernel Name\s+(.+?)\s*$")
_SOURCE_LINE = re.compile(r"^\s*(\d+)\s{2,}(.*?)\s*$")
_INSTRUCTION = re.compile(r"^\s*(0x[0-9a-fA-F]+)\s+(.+?)\s*$")
_NUMBER_WITH_UNIT = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$"
)
_DETAIL_FIELDS = (
    "Kernel ID",
    "Process ID",
    "Process Name",
    "Kernel Name",
    "Context",
    "Stream",
    "Section",
    "Metrics",
    "Value",
)


def _clean_line(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value.rstrip("\r\n"))


def _normalize_source_path(
    reported: str,
    source_paths: Mapping[str, str],
) -> str:
    normalized_reported = Path(reported).as_posix()
    normalized_sources = {
        Path(absolute).as_posix(): relative
        for absolute, relative in source_paths.items()
    }
    if normalized_reported in normalized_sources:
        return normalized_sources[normalized_reported]
    suffix_matches = [
        relative
        for absolute, relative in normalized_sources.items()
        if normalized_reported.endswith("/" + relative)
        or absolute.endswith("/" + Path(normalized_reported).name)
    ]
    return suffix_matches[0] if len(set(suffix_matches)) == 1 else normalized_reported


def parse_ixkn_source_mapping(
    text: str,
    *,
    source_paths: Mapping[str, str],
    expected_kernel: str = "",
) -> Dict[str, Any]:
    """Parse IXKN's cuda,assembly page into source-line mappings."""
    mappings: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    current_line: Dict[str, Any] | None = None

    def finish_current() -> None:
        nonlocal current, current_line
        if current is not None and current.get("lines"):
            current["mapped_line_count"] = sum(
                1
                for line in current["lines"]
                if line["source"] != "<Unable to open source file>"
                and line["instructions"]
            )
            current["instruction_count"] = sum(
                len(line["instructions"]) for line in current["lines"]
            )
            mappings.append(current)
        current = None
        current_line = None

    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        file_match = _FILE_PATH.match(line)
        if file_match:
            finish_current()
            reported_path = file_match.group(1)
            current = {
                "kernel_name": expected_kernel,
                "function_name": "",
                "source_path": _normalize_source_path(reported_path, source_paths),
                "reported_source_path": reported_path,
                "lines": [],
            }
            continue
        function_match = _FUNCTION_NAME.match(line)
        if function_match:
            if current is None:
                current = {
                    "kernel_name": expected_kernel,
                    "function_name": "",
                    "source_path": "",
                    "reported_source_path": "",
                    "lines": [],
                }
            function_name = function_match.group(1)
            current["function_name"] = function_name
            if not current["kernel_name"]:
                current["kernel_name"] = function_name
            continue
        if current is None or not current.get("function_name"):
            continue
        instruction_match = _INSTRUCTION.match(line)
        if instruction_match and current_line is not None:
            current_line["instructions"].append(
                {
                    "address": instruction_match.group(1),
                    "text": instruction_match.group(2),
                }
            )
            continue
        source_match = _SOURCE_LINE.match(line)
        if source_match:
            current_line = {
                "line_number": int(source_match.group(1)),
                "source": source_match.group(2),
                "instructions": [],
            }
            current["lines"].append(current_line)

    finish_current()
    if not mappings:
        raise ValueError("IXKN source page contained no source-to-assembly mapping")
    return {
        "schema_version": 1,
        "profiler": "ixkn",
        "kernels": mappings,
        "summary": {
            "kernel_count": len(mappings),
            "source_file_count": len(
                {item["source_path"] for item in mappings if item["source_path"]}
            ),
            "mapped_source_line_count": sum(
                item["mapped_line_count"] for item in mappings
            ),
            "mapped_instruction_count": sum(
                item["instruction_count"] for item in mappings
            ),
        },
    }


def merge_ixkn_source_mappings(
    reports: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    kernels = [
        dict(kernel) for report in reports for kernel in report.get("kernels", [])
    ]
    return {
        "schema_version": 1,
        "profiler": "ixkn",
        "kernels": kernels,
        "summary": {
            "kernel_count": len(kernels),
            "source_file_count": len(
                {item["source_path"] for item in kernels if item["source_path"]}
            ),
            "mapped_source_line_count": sum(
                int(item.get("mapped_line_count", 0)) for item in kernels
            ),
            "mapped_instruction_count": sum(
                int(item.get("instruction_count", 0)) for item in kernels
            ),
        },
    }


def render_ixkn_source_details(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Section: Iluvatar IXKN Source Mapping",
        f"Profiled Kernels: {summary['kernel_count']}",
        f"Source Files: {summary['source_file_count']}",
        f"Mapped Source Lines: {summary['mapped_source_line_count']}",
        f"Mapped Instructions: {summary['mapped_instruction_count']}",
    ]
    for kernel in report["kernels"]:
        lines.extend(
            [
                "",
                f"Kernel: {kernel['kernel_name']}",
                f"Function: {kernel['function_name']}",
                f"Source: {kernel['source_path']}",
            ]
        )
        for source_line in kernel["lines"]:
            instructions = source_line["instructions"]
            if instructions:
                instruction_text = "; ".join(
                    f"{item['address']} {item['text']}" for item in instructions
                )
                lines.append(
                    f"{source_line['line_number']}: {source_line['source']} | "
                    f"{instruction_text}"
                )
    return "\n".join(lines) + "\n"


def write_ixkn_source_mapping(
    report: Mapping[str, Any],
    path: Path,
) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_ixkn_assembly_listing(
    text: str,
    *,
    expected_kernel: str = "",
) -> Dict[str, Any]:
    """Parse IXKN's assembly-only source page when CUDA line info is unavailable."""
    kernel_name = expected_kernel
    instructions: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        kernel_match = _KERNEL_NAME.match(line)
        if kernel_match and not kernel_name:
            kernel_name = kernel_match.group(1).strip()
            continue
        instruction_match = _INSTRUCTION.match(line)
        if instruction_match is None:
            continue
        instruction_text = instruction_match.group(2).strip()
        opcode, _, operands = instruction_text.partition(" ")
        instructions.append(
            {
                "kernel_name": kernel_name,
                "function_name": kernel_name,
                "pc": instruction_match.group(1),
                "opcode": opcode,
                "operands": operands.strip(),
                "text": instruction_text,
                "source_file": "",
                "source_line": None,
                "source_text": "",
            }
        )
    if not instructions:
        raise ValueError("IXKN assembly page contained no instructions")
    return {
        "schema_version": 1,
        "profiler": "ixkn",
        "evidence_type": "static_instruction_listing",
        "dynamic_timeline": False,
        "instruction_count": len(instructions),
        "mapped_instruction_count": 0,
        "instructions": instructions,
    }


def merge_ixkn_instruction_listings(
    listings: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Merge per-kernel static instruction listings without losing mapping fields."""
    instructions = [
        dict(instruction)
        for listing in listings
        for instruction in listing.get("instructions", [])
    ]
    return {
        "schema_version": 1,
        "profiler": "ixkn",
        "evidence_type": "static_instruction_listing",
        "dynamic_timeline": False,
        "instruction_count": len(instructions),
        "mapped_instruction_count": sum(
            1 for instruction in instructions if instruction.get("source_file")
        ),
        "instructions": instructions,
    }


def build_ixkn_instruction_listing(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten IXKN source/assembly evidence into a queryable instruction list."""
    instructions: List[Dict[str, Any]] = []
    mapped_instruction_count = 0
    for kernel in report.get("kernels", []):
        for source_line in kernel.get("lines", []):
            source_text = str(source_line.get("source", ""))
            source_file = str(kernel.get("source_path", ""))
            source_line_number = source_line.get("line_number")
            mapped = bool(
                source_file
                and source_line_number is not None
                and source_text != "<Unable to open source file>"
            )
            for instruction in source_line.get("instructions", []):
                text = str(instruction.get("text", "")).strip()
                opcode, _, operands = text.partition(" ")
                instructions.append(
                    {
                        "kernel_name": kernel.get("kernel_name", ""),
                        "function_name": kernel.get("function_name", ""),
                        "pc": instruction.get("address", ""),
                        "opcode": opcode,
                        "operands": operands.strip(),
                        "text": text,
                        "source_file": source_file if mapped else "",
                        "source_line": source_line_number if mapped else None,
                        "source_text": source_text if mapped else "",
                    }
                )
                mapped_instruction_count += int(mapped)
    return {
        "schema_version": 1,
        "profiler": "ixkn",
        "evidence_type": "static_instruction_listing",
        "dynamic_timeline": False,
        "instruction_count": len(instructions),
        "mapped_instruction_count": mapped_instruction_count,
        "instructions": instructions,
    }


def render_ixkn_instruction_summary(
    listing: Mapping[str, Any],
    counter_report: Mapping[str, Any] | None = None,
) -> str:
    """Render concise static instruction coverage plus dynamic counter totals."""
    instructions = list(listing.get("instructions", []))
    opcode_counts = Counter(
        str(instruction.get("opcode", "")) for instruction in instructions
    )
    lines = [
        "Iluvatar IXKN Instruction Summary",
        "==================================",
        f"Instructions: {listing.get('instruction_count', 0)}",
        f"Source-mapped Instructions: {listing.get('mapped_instruction_count', 0)}",
        "Dynamic Instruction Timeline: unavailable",
    ]
    if counter_report is not None:
        summary = counter_report.get("summary", {})
        lines.extend(
            [
                f"Counter Records: {summary.get('metric_count', 0)}",
                f"Executed Instructions: {summary.get('executed_instruction_count', 0)}",
            ]
        )
    lines.extend(["", "Top Opcodes"])
    for opcode, count in opcode_counts.most_common(20):
        lines.append(f"  {opcode}: {count}")
    if not opcode_counts:
        lines.append("  No instructions were parsed.")
    return "\n".join(lines).rstrip() + "\n"


def _split_detail_line(line: str) -> List[str]:
    line = _clean_line(line).strip()
    if not line:
        return []
    if line.startswith('"') and line.endswith('"'):
        # IXKN surrounds every field with quotes but does not escape quotes in
        # Process Name. Splitting only on the complete field delimiter retains
        # embedded Python strings and commas verbatim.
        return [value.replace('""', '"') for value in line[1:-1].split('","')]
    return next(csv.reader([line]))


def _repair_detail_row(row: Sequence[str]) -> Dict[str, str] | None:
    if len(row) < len(_DETAIL_FIELDS):
        return None
    if len(row) == len(_DETAIL_FIELDS):
        values = list(row)
    else:
        values = [
            row[0],
            row[1],
            ",".join(row[2:-6]),
            *row[-6:],
        ]
    return dict(zip(_DETAIL_FIELDS, values))


def _numeric_value(value: str) -> Tuple[float | None, str]:
    match = _NUMBER_WITH_UNIT.match(value)
    if match is None:
        return None, ""
    try:
        return float(match.group(1)), match.group(2).strip()
    except ValueError:
        return None, ""


def parse_ixkn_details_csv(text: str) -> Tuple[Dict[str, Any], List[str]]:
    """Parse IXKN details CSV, including its malformed Process Name quoting."""
    rows = [_split_detail_line(line) for line in text.splitlines()]
    warnings: List[str] = []
    header_index = next(
        (index for index, row in enumerate(rows) if tuple(row) == _DETAIL_FIELDS),
        None,
    )
    if header_index is None:
        raise ValueError("IXKN details output did not contain the expected CSV header")

    parsed_rows: List[Dict[str, Any]] = []
    for index, raw_row in enumerate(rows[header_index + 1 :], start=2):
        if not raw_row or not any(value.strip() for value in raw_row):
            continue
        row = _repair_detail_row(raw_row)
        if row is None:
            warnings.append(f"ignored malformed IXKN details row {index}")
            continue
        metric = row["Metrics"].strip()
        raw_value = row["Value"].strip()
        numeric, unit = _numeric_value(raw_value)
        record: Dict[str, Any] = {
            "kernel_id": row["Kernel ID"].strip(),
            "process_id": row["Process ID"].strip(),
            "process_name": row["Process Name"].strip(),
            "kernel_name": row["Kernel Name"].strip(),
            "context": row["Context"].strip(),
            "stream": row["Stream"].strip(),
            "section": row["Section"].strip(),
            "metric": metric,
            "value": raw_value,
            "record_type": "message" if metric in {"INF", "WRN", "ERR"} else "metric",
        }
        if metric in {"INF", "WRN", "ERR"}:
            record["severity"] = {
                "INF": "info",
                "WRN": "warning",
                "ERR": "error",
            }[metric]
        if numeric is not None:
            record["numeric_value"] = numeric
            record["unit"] = unit
        parsed_rows.append(record)

    if not parsed_rows:
        raise ValueError("IXKN details output contained no records")

    kernel_index: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    section_index: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = {}
    for record in parsed_rows:
        kernel_key = (
            record["kernel_id"],
            record["process_id"],
            record["kernel_name"],
            record["context"],
            record["stream"],
        )
        kernel = kernel_index.get(kernel_key)
        if kernel is None:
            kernel = {
                "kernel_id": record["kernel_id"],
                "process_id": record["process_id"],
                "process_name": record["process_name"],
                "kernel_name": record["kernel_name"],
                "context": record["context"],
                "stream": record["stream"],
                "sections": [],
            }
            kernel_index[kernel_key] = kernel
        section_key = (kernel_key, record["section"])
        section = section_index.get(section_key)
        if section is None:
            section = {"name": record["section"], "records": []}
            section_index[section_key] = section
            kernel["sections"].append(section)
        section["records"].append(
            {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "kernel_id",
                    "process_id",
                    "process_name",
                    "kernel_name",
                    "context",
                    "stream",
                    "section",
                }
            }
        )

    order = {name: index for index, name in enumerate(IXKN_SECTION_ORDER)}
    kernels = list(kernel_index.values())
    for kernel in kernels:
        kernel["sections"].sort(
            key=lambda section: (
                order.get(section["name"], len(order)),
                section["name"],
            )
        )
    section_names = sorted(
        {record["section"] for record in parsed_rows},
        key=lambda name: (order.get(name, len(order)), name),
    )
    instruction_count = sum(
        int(record["numeric_value"])
        for record in parsed_rows
        if record["metric"] == "Executed Instructions" and "numeric_value" in record
    )
    report = {
        "schema_version": 1,
        "profiler": "ixkn",
        "section_names": section_names,
        "kernels": kernels,
        "summary": {
            "kernel_count": len(kernels),
            "section_count": len(section_names),
            "record_count": len(parsed_rows),
            "metric_count": sum(
                record["record_type"] == "metric" for record in parsed_rows
            ),
            "message_count": sum(
                record["record_type"] == "message" for record in parsed_rows
            ),
            "warning_count": sum(
                record.get("severity") == "warning" for record in parsed_rows
            ),
            "executed_instruction_count": instruction_count,
        },
    }
    return report, warnings


def iter_ixkn_instruction_rows(
    report: Mapping[str, Any],
) -> Iterable[Dict[str, Any]]:
    for kernel in report["kernels"]:
        for section in kernel["sections"]:
            for record in section["records"]:
                yield {
                    "kernel_id": kernel["kernel_id"],
                    "kernel_name": kernel["kernel_name"],
                    "context": kernel["context"],
                    "stream": kernel["stream"],
                    "section": section["name"],
                    "metric": record["metric"],
                    "value": record["value"],
                    "numeric_value": record.get("numeric_value", ""),
                    "unit": record.get("unit", ""),
                    "record_type": record["record_type"],
                    "severity": record.get("severity", ""),
                }


def write_ixkn_instruction_artifacts(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fieldnames = [
        "kernel_id",
        "kernel_name",
        "context",
        "stream",
        "section",
        "metric",
        "value",
        "numeric_value",
        "unit",
        "record_type",
        "severity",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(iter_ixkn_instruction_rows(report))


def merge_ixkn_instruction_reports(
    reports: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    reports = list(reports)
    kernels = [
        dict(kernel) for report in reports for kernel in report.get("kernels", [])
    ]
    section_names = [
        name
        for name in IXKN_SECTION_ORDER
        if any(name in report.get("section_names", []) for report in reports)
    ]
    extra_sections = sorted(
        {
            name
            for report in reports
            for name in report.get("section_names", [])
            if name not in IXKN_SECTION_ORDER
        }
    )
    section_names.extend(extra_sections)
    summary_keys = (
        "record_count",
        "metric_count",
        "message_count",
        "warning_count",
        "executed_instruction_count",
    )
    return {
        "schema_version": 1,
        "profiler": "ixkn",
        "section_names": section_names,
        "kernels": kernels,
        "summary": {
            "kernel_count": len(kernels),
            "section_count": len(section_names),
            **{
                key: sum(int(report["summary"].get(key, 0)) for report in reports)
                for key in summary_keys
            },
        },
    }


def render_ixkn_instruction_details(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Section: Iluvatar IXKN Detailed Counter Report",
        f"Profiled Kernels: {summary['kernel_count']}",
        f"Sections: {summary['section_count']}",
        f"Metric Records: {summary['metric_count']}",
        f"Diagnostic Messages: {summary['message_count']}",
        f"Warning Messages: {summary['warning_count']}",
        f"Executed Instructions: {summary['executed_instruction_count']}",
    ]
    for kernel in report["kernels"]:
        lines.extend(["", f"Kernel: {kernel['kernel_name']}"])
        for section in kernel["sections"]:
            lines.append(f"[{section['name']}]")
            for record in section["records"]:
                prefix = record.get("severity", record["metric"])
                lines.append(f"{prefix}: {record['value']}")
    return "\n".join(lines) + "\n"
