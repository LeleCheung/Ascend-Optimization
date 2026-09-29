# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Parse and render Ascend AI Core simulator instruction evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ...protocol.version import KERNELGEN_API_VERSION

def _as_int(value: Any) -> Optional[int]:
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pc(value: Any) -> str:
    parsed = _as_int(value)
    return f"0x{parsed:x}" if parsed is not None else str(value or "")


def _source_locations(value: Any) -> List[Dict[str, Any]]:
    locations: List[Dict[str, Any]] = []
    for raw_location in str(value or "").splitlines():
        path, separator, raw_line = raw_location.strip().rpartition(":")
        if not separator or not path:
            continue
        line = _as_int(raw_line)
        if line is None:
            continue
        locations.append({"file": path, "line": line})
    return locations


def _source_index(source_files: Optional[Mapping[str, str]]) -> Dict[str, Tuple[str, List[str]]]:
    index: Dict[str, Tuple[str, List[str]]] = {}
    basename_counts: Dict[str, int] = {}
    if not source_files:
        return index
    for path in source_files:
        basename = Path(path).name
        basename_counts[basename] = basename_counts.get(basename, 0) + 1
    for path, content in source_files.items():
        normalized = Path(path).as_posix()
        entry = (normalized, content.splitlines())
        index[normalized] = entry
        if basename_counts[Path(path).name] == 1:
            index[Path(path).name] = entry
    return index


def _enrich_location(
    location: Dict[str, Any], source_index: Mapping[str, Tuple[str, List[str]]]
) -> Dict[str, Any]:
    raw_path = str(location["file"])
    entry = source_index.get(raw_path)
    if entry is None:
        for key, candidate in source_index.items():
            if "/" in key and raw_path.endswith(f"/{key}"):
                entry = candidate
                break
    if entry is None:
        entry = source_index.get(Path(raw_path).name)
    if entry is None:
        return dict(location)
    path, lines = entry
    enriched: Dict[str, Any] = {"file": path, "line": location["line"]}
    line_number = int(location["line"])
    if 0 < line_number <= len(lines):
        enriched["code"] = lines[line_number - 1].strip()
    return enriched


def _enrich_locations(
    locations: Iterable[Dict[str, Any]], source_index: Mapping[str, Tuple[str, List[str]]]
) -> List[Dict[str, Any]]:
    return [_enrich_location(location, source_index) for location in locations]


def _primary_source(locations: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not locations:
        return None
    for location in reversed(locations):
        if int(location.get("line", 0)) > 0 and not str(location.get("file", "")).endswith(
            "/internal"
        ):
            return location
    return locations[-1]


def _read_trace(path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"Could not parse simulator trace {path.name}: {exc}"]
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else []
    pipeline_by_pid: Dict[Any, str] = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            pipeline_by_pid[event.get("pid")] = str(event.get("args", {}).get("name", ""))

    by_pc: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args")
        if not isinstance(args, dict) or "pc_addr" not in args:
            continue
        pipeline = pipeline_by_pid.get(event.get("pid"), "")
        if pipeline == "CACHEMISS":
            continue
        pc = _pc(args.get("pc_addr"))
        candidate = {
            "instruction": str(event.get("name", "")),
            "pipeline": pipeline,
            "detail": str(args.get("detail", "")),
            "source_locations": _source_locations(args.get("code")),
        }
        current = by_pc.get(pc)
        if current is None or (not current["source_locations"] and candidate["source_locations"]):
            by_pc[pc] = candidate
    return by_pc, warnings


def _read_instruction_csv(
    path: Path,
    trace_by_pc: Mapping[str, Dict[str, Any]],
    source_index: Mapping[str, Tuple[str, List[str]]],
) -> List[Dict[str, Any]]:
    instructions: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            pc = _pc(row.get("addr"))
            trace = trace_by_pc.get(pc, {})
            item: Dict[str, Any] = {
                "pc": pc,
                "instruction": str(row.get("instr") or trace.get("instruction") or ""),
                "pipeline": str(row.get("pipe") or trace.get("pipeline") or ""),
                "calls": _as_int(row.get("call_count")) or 0,
                "cycles": _as_int(row.get("cycles")) or 0,
                "running_time_us": _as_float(row.get("running_time(us)")) or 0.0,
                "detail": str(row.get("detail") or trace.get("detail") or ""),
            }
            locations = _enrich_locations(trace.get("source_locations", []), source_index)
            source = _primary_source(locations)
            if source is not None:
                item["source"] = source
            if len(locations) > 1:
                item["source_stack"] = locations
            instructions.append(item)
    return sorted(instructions, key=lambda item: item["cycles"], reverse=True)


def _read_source_csv(
    path: Path, source_index: Mapping[str, Tuple[str, List[str]]]
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            locations = _enrich_locations(_source_locations(row.get("code")), source_index)
            source = _primary_source(locations)
            item: Dict[str, Any] = {
                "calls": _as_int(row.get("call_count")) or 0,
                "cycles": _as_int(row.get("cycles")) or 0,
                "running_time_us": _as_float(row.get("running_time(us)")) or 0.0,
            }
            if source is not None:
                item.update(source)
            if len(locations) > 1:
                item["source_stack"] = locations
            summary.append(item)
    return sorted(summary, key=lambda item: item["cycles"], reverse=True)


def build_instruction_report(
    output_root: Path,
    kernel_names: Sequence[str],
    soc_version: str,
    source_files: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Build an agent-readable report from public msprof simulator exports."""
    warnings: List[str] = []
    kernels: List[Dict[str, Any]] = []
    source_index = _source_index(source_files)
    for index, kernel_name in enumerate(kernel_names):
        kernel_root = output_root / f"kernel-{index:03d}"
        simulator_roots = sorted(kernel_root.glob("OPPROF_*/simulator"))
        if not simulator_roots:
            warnings.append(f"Simulator report for kernel {kernel_name!r} has no simulator data")
            continue
        simulator_root = simulator_roots[-1]
        cores: List[Dict[str, Any]] = []
        for instruction_csv in sorted(simulator_root.glob("*/*_instr_exe.csv")):
            core_name = instruction_csv.parent.name
            trace_by_pc, trace_warnings = _read_trace(instruction_csv.parent / "trace.json")
            warnings.extend(trace_warnings)
            try:
                instructions = _read_instruction_csv(instruction_csv, trace_by_pc, source_index)
            except Exception as exc:
                warnings.append(
                    f"Could not parse simulator instructions for {kernel_name!r}/{core_name}: {exc}"
                )
                continue
            source_csv = instruction_csv.with_name(
                instruction_csv.name.replace("_instr_exe.csv", "_code_exe.csv")
            )
            source_summary: List[Dict[str, Any]] = []
            if source_csv.is_file():
                try:
                    source_summary = _read_source_csv(source_csv, source_index)
                except Exception as exc:
                    warnings.append(
                        f"Could not parse simulator source summary for {kernel_name!r}/{core_name}: {exc}"
                    )
            cores.append(
                {
                    "name": core_name,
                    "listed_instruction_count": len(instructions),
                    "mapped_instruction_count": sum("source" in item for item in instructions),
                    "instructions": instructions,
                    "source_summary": source_summary,
                }
            )
        if cores:
            kernels.append({"name": kernel_name, "cores": cores})
        else:
            warnings.append(f"Simulator report for kernel {kernel_name!r} has no instruction CSV")
    report = {
        "api_version": KERNELGEN_API_VERSION,
        "backend": "npu",
        "profiler": "msprof-op-simulator",
        "soc_version": soc_version,
        "kernels": kernels,
    }
    return report, warnings


def render_instruction_details(report: Mapping[str, Any]) -> str:
    """Render simulator evidence as factual text without optimization advice."""
    lines = [
        "Section: AI Core Simulator Instruction Profile",
        f"SoC Version: {report.get('soc_version', '')}",
        f"Kernel Count: {len(report.get('kernels', []))}",
        "Timing Scope: simulator instruction execution; eval latency remains authoritative",
    ]
    for kernel_index, kernel in enumerate(report.get("kernels", []), 1):
        lines.extend(["", f"Kernel {kernel_index}: {kernel.get('name', '')}"])
        for core in kernel.get("cores", []):
            lines.extend(
                [
                    f"Core: {core.get('name', '')}",
                    f"Listed Instructions: {core.get('listed_instruction_count', 0)}",
                    f"Mapped Instructions: {core.get('mapped_instruction_count', 0)}",
                    "Source Hotspots:",
                    "Source | Calls | Cycles | Running Time(us)",
                ]
            )
            for source in core.get("source_summary", []):
                location = f"{source.get('file', '')}:{source.get('line', '')}"
                if source.get("code"):
                    location = f"{location} {source['code']}"
                lines.append(
                    f"{location} | {source.get('calls', 0)} | {source.get('cycles', 0)} | {source.get('running_time_us', 0.0):.6f}"
                )
            lines.extend(
                [
                    "Instruction Listing:",
                    "PC | Instruction | Pipeline | Calls | Cycles | Running Time(us) | Source",
                ]
            )
            for instruction in core.get("instructions", []):
                source = instruction.get("source", {})
                location = ""
                if source:
                    location = f"{source.get('file', '')}:{source.get('line', '')}"
                lines.append(
                    f"{instruction.get('pc', '')} | {instruction.get('instruction', '')} | {instruction.get('pipeline', '')} | {instruction.get('calls', 0)} | {instruction.get('cycles', 0)} | {instruction.get('running_time_us', 0.0):.6f} | {location}"
                )
    return "\n".join(lines) + "\n"
