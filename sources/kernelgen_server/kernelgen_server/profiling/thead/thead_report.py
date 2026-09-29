# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""ACU CSV, source, and instruction report normalization."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..source_view import ReportSourceTarget


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
        return "    " + "  ".join(
            values[index].ljust(widths[index]) for index in range(len(widths))
        ).rstrip()

    output = [line(columns), "    " + "  ".join("-" * width for width in widths)]
    output.extend(line(row) for row in normalized)
    return output


class THeadReportMixin:
    @staticmethod
    def _parse_value(raw: str) -> Any:
        text = raw.strip().replace(",", "")
        if not text or text.lower() in {"n/a", "nan"}:
            return None
        try:
            value = float(text)
        except ValueError:
            return raw.strip()
        return int(value) if value.is_integer() else value

    @staticmethod
    def _first(row: dict[str, str], *names: str) -> str:
        for name in names:
            value = row.get(name, "")
            if value:
                return value
        return ""

    def _parse_report_csv(self, csv_path: Path) -> Dict[str, Any]:
        """Preserve launch, kernel, section and unit context from ACU CSV."""
        kernels: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "ID" not in reader.fieldnames:
                raise ValueError("ACU CSV is missing the ID header")
            for row in reader:
                metric_name = self._first(row, "Metric Name").strip()
                section_name = self._first(row, "Section Name").strip()
                if not metric_name or not section_name:
                    continue
                key = (
                    self._first(row, "ID", "Launch ID"),
                    self._first(row, "Device"),
                    self._first(row, "Kernel Mangled Name"),
                    self._first(row, "Kernel Name"),
                )
                kernel = kernels.setdefault(
                    key,
                    {
                        "id": key[0],
                        "name": key[3],
                        "mangled_name": key[2],
                        "device": key[1],
                        "process_id": self._first(row, "Process ID"),
                        "context": self._first(row, "Context"),
                        "stream": self._first(row, "Stream"),
                        "grid_size": self._first(row, "Grid Size"),
                        "block_size": self._first(row, "Block Size"),
                        "sections": [],
                    },
                )
                section = next(
                    (item for item in kernel["sections"] if item["name"] == section_name),
                    None,
                )
                if section is None:
                    section = {"name": section_name, "metrics": []}
                    kernel["sections"].append(section)
                section["metrics"].append(
                    {
                        "name": metric_name,
                        "unit": self._first(row, "Metric Unit"),
                        "value": self._parse_value(self._first(row, "Metric Value")),
                    }
                )
        return {"kernels": list(kernels.values())}

    @staticmethod
    def _render_profile_details(
        metrics: Dict[str, Any],
        summary: Dict[str, Any],
        *,
        device: str,
        warmup: int,
        iterations: int,
    ) -> str:
        """Render normalized ACU metrics without duplicating the artifact manifest."""
        lines = [
            "PPU ACU Metrics Summary",
            "=======================",
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
                    ("Timing Authority", "diagnostic only; evaluator latency is authoritative"),
                ],
            )
        )
        lines.extend(["", "Section: Kernel Summary", ""])
        kernels = metrics.get("kernels", [])
        if kernels:
            lines.extend(
                _render_table(
                    ["ID", "Kernel", "Device", "Grid", "Block", "Sections"],
                    (
                        (
                            kernel.get("id", ""),
                            kernel.get("name", ""),
                            kernel.get("device", ""),
                            kernel.get("grid_size", ""),
                            kernel.get("block_size", ""),
                            len(kernel.get("sections", [])),
                        )
                        for kernel in kernels
                    ),
                )
            )
        else:
            lines.append("    No kernel records were parsed.")

        lines.extend(["", "Section: Performance Signals", ""])
        signal_rows = []
        for kernel in kernels:
            for section in kernel.get("sections", []):
                for metric in section.get("metrics", []):
                    signal_rows.append(
                        (
                            kernel.get("name", ""),
                            section.get("name", ""),
                            metric.get("name", ""),
                            metric.get("value", ""),
                            metric.get("unit", ""),
                        )
                    )
        if signal_rows:
            lines.extend(
                _render_table(
                    ["Kernel", "Section", "Metric", "Value", "Unit"], signal_rows
                )
            )
        else:
            lines.append("    ACU exported no numeric performance signals.")
        return "\n".join(lines).rstrip() + "\n"
    @staticmethod
    def _iter_binaries(cache_dir: Path) -> Iterable[Path]:
        for suffix in ("*.cubin", "*.hsaco", "*.elf", "*.bin"):
            yield from cache_dir.rglob(suffix)

    @staticmethod
    def _parse_objdump(text: str, binary: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        source_rows: list[dict[str, Any]] = []
        instructions: list[dict[str, Any]] = []
        current_source = ""
        current_line: int | None = None
        section = ""
        source_re = re.compile(r"^;\s+(.+):(\d+)\s*$")
        label_re = re.compile(r"^\s*[0-9a-fA-F]+\s+<[^>]+>:$")
        section_re = re.compile(r"^Disassembly of section\s+(.+):$")
        instruction_re = re.compile(
            r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}\s+)+\s*([^\s]+)(?:\s+(.*?))?\s*$"
        )
        seen_sources: set[tuple[str, int]] = set()
        for line in text.splitlines():
            section_match = section_re.match(line)
            if section_match:
                section = section_match.group(1)
                current_source = ""
                current_line = None
                continue
            source_match = source_re.match(line)
            if source_match:
                current_source = source_match.group(1)
                current_line = int(source_match.group(2))
                source_key = (current_source, current_line)
                if source_key not in seen_sources:
                    source_rows.append(
                        {"binary": binary.name, "file": current_source, "line": current_line}
                    )
                    seen_sources.add(source_key)
                continue
            if label_re.match(line):
                current_source = ""
                current_line = None
                continue
            instruction_match = instruction_re.match(line)
            # The plain .text section is ELF padding/stubs (typically only NOPs).
            # Real PPU kernels are emitted in named .text.* sections.
            if not instruction_match or not section.startswith(".text."):
                continue
            function = section.removeprefix(".text.kernel.").removeprefix(".text.")
            instructions.append(
                {
                    "binary": binary.name,
                    "section": section,
                    "function": function,
                    "pc": f"0x{instruction_match.group(1).lower()}",
                    "opcode": instruction_match.group(2),
                    "operands": (instruction_match.group(3) or "").strip(),
                    "source_file": current_source or None,
                    "source_line": current_line,
                }
            )
        return source_rows, instructions

    @staticmethod
    def _normalize_source_evidence(
        source_rows: list[dict[str, Any]],
        instructions: list[dict[str, Any]],
        target: ReportSourceTarget,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        source_contents = {
            source.path: source.content.splitlines() for source in target.implementation.sources
        }

        def resolve(raw_path: str, line_number: int | None) -> tuple[str, str | None]:
            normalized = raw_path.replace("\\", "/")
            matches = [
                path
                for path in source_contents
                if normalized == path
                or normalized.endswith(f"/source/{path}")
                or normalized.endswith(f"/{path}")
            ]
            path = max(matches, key=len) if matches else normalized
            lines = source_contents.get(path, [])
            text = lines[line_number - 1] if line_number and line_number <= len(lines) else None
            return path, text

        normalized_sources: list[dict[str, Any]] = []
        for row in source_rows:
            path, source_text = resolve(row["file"], row["line"])
            normalized_sources.append(
                {
                    **row,
                    "file": path,
                    "source_text": source_text,
                }
            )

        normalized_instructions: list[dict[str, Any]] = []
        for row in instructions:
            raw_path = row.get("source_file")
            if raw_path:
                path, source_text = resolve(raw_path, row.get("source_line"))
            else:
                path, source_text = None, None
            normalized_instructions.append(
                {
                    **row,
                    "source_file": path,
                    "source_text": source_text,
                }
            )
        return normalized_sources, normalized_instructions

    @staticmethod
    def _source_mapping_payload(
        sources: list[dict[str, Any]],
        instructions: list[dict[str, Any]],
        target: ReportSourceTarget,
    ) -> Dict[str, Any]:
        locations: dict[tuple[str, str, int], dict[str, Any]] = {}
        for source in sources:
            key = (source["binary"], source["file"], source["line"])
            locations[key] = {
                "binary": source["binary"],
                "file": source["file"],
                "line": source["line"],
                "source_text": source.get("source_text"),
                "instructions": [],
            }
        for instruction in instructions:
            file = instruction.get("source_file")
            line = instruction.get("source_line")
            if not file or line is None:
                continue
            key = (instruction["binary"], file, line)
            location = locations.setdefault(
                key,
                {
                    "binary": instruction["binary"],
                    "file": file,
                    "line": line,
                    "source_text": instruction.get("source_text"),
                    "instructions": [],
                },
            )
            location["instructions"].append(
                {
                    "section": instruction["section"],
                    "function": instruction["function"],
                    "pc": instruction["pc"],
                    "opcode": instruction["opcode"],
                }
            )
        return {
            "evidence_type": "static_source_mapping",
            "dynamic_hotspots": False,
            "source_files": [
                {"path": source.path, "content": source.content}
                for source in target.implementation.sources
            ],
            "source_locations": list(locations.values()),
        }

    @staticmethod
    def _render_source_report(payload: Dict[str, Any]) -> str:
        lines = [
            "PPU static source mapping",
            "Dynamic source hotspots: unavailable",
            "",
        ]
        for location in payload["source_locations"]:
            lines.append(
                f"{location['file']}:{location['line']}  {location.get('source_text') or ''}".rstrip()
            )
            for instruction in location["instructions"]:
                lines.append(
                    f"  {instruction['pc']:>10}  {instruction['opcode']:<28} "
                    f"[{instruction['function']}]"
                )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _render_instruction_report(instructions: list[dict[str, Any]]) -> str:
        lines = [
            "PPU static instruction listing",
            "Dynamic instruction timeline: unavailable",
            "",
            "binary\tfunction\tpc\topcode\toperands\tsource",
        ]
        for instruction in instructions:
            source = ""
            if instruction.get("source_file") and instruction.get("source_line") is not None:
                source = f"{instruction['source_file']}:{instruction['source_line']}"
            lines.append(
                "\t".join(
                    [
                        instruction["binary"],
                        instruction["function"],
                        instruction["pc"],
                        instruction["opcode"],
                        instruction["operands"],
                        source,
                    ]
                )
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_instruction_summary(
        instructions: list[dict[str, Any]],
        *,
        binary_count: int,
        disassembled_binary_count: int,
    ) -> str:
        mapped = [
            instruction
            for instruction in instructions
            if instruction.get("source_file") and instruction.get("source_line") is not None
        ]
        opcode_counts = Counter(instruction.get("opcode", "") for instruction in instructions)
        lines = [
            "PPU Static Instruction Summary",
            "==============================",
            f"Binaries Found: {binary_count}",
            f"Binaries Disassembled: {disassembled_binary_count}",
            f"Instructions: {len(instructions)}",
            f"Source-mapped Instructions: {len(mapped)}",
            "Dynamic Instruction Timeline: unavailable",
            "",
            "Top Opcodes",
        ]
        for opcode, count in opcode_counts.most_common(20):
            lines.append(f"  {opcode}: {count}")
        if not opcode_counts:
            lines.append("  No instructions were parsed.")
        return "\n".join(lines).rstrip() + "\n"
