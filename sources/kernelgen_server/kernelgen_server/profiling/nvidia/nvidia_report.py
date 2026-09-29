# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Parse Nsight Compute reports into agent-readable metrics and SASS data."""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class NcuReportError(RuntimeError):
    """Raised when an NCU report cannot provide the requested evidence."""


@dataclass(frozen=True)
class NcuAnalysis:
    text: str
    metrics: dict[str, Any]
    sass: dict[str, Any]
    instruction_count: int
    mapped_instruction_count: int


@dataclass
class _SassInstr:
    pc: int
    sass: str
    ptx: str
    file: str
    line: int
    exec_count: float = 0.0
    stalls: dict[str, float] = field(default_factory=dict)
    live_regs: int = 0


_STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
_NON_STALL_REASONS = {"selected", "not_selected"}
_BACKWARD_ATTRIBUTE_REASONS = {"long_scoreboard", "short_scoreboard"}
_LONG_LATENCY_OPS = {"LDG", "LDL", "RED", "ATOM", "ATOMS"}
_MIO_OPS = {"S2R", "CS2R", "LDS", "STS", "LDSM", "LDC", "ULDC"}
_WIDE_WRITE_OPS = {
    "LDG.E.128": 4,
    "STG.E.128": 0,
    "LDG.E.64": 2,
    "LDG.E": 1,
    "LDC.64": 2,
    "ULDC.64": 0,
    "IMAD.WIDE": 2,
}
_REG_PATTERN = re.compile(r"\bR(\d+)\b")


def _import_ncu_report():
    try:
        import ncu_report as module

        return module
    except ImportError:
        pass

    candidates: list[Path] = []
    for variable in ("NCU_REPORT_PYTHON_PATH", "NSIGHT_COMPUTE_PYTHON"):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]))

    executable = shutil.which("ncu")
    if executable:
        for parent in Path(executable).resolve().parents:
            candidate = parent / "extras" / "python"
            if candidate.is_dir():
                candidates.append(candidate)
                break

    install_root = Path("/opt/nvidia/nsight-compute")
    if install_root.is_dir():
        candidates.extend(
            sorted(
                (
                    directory / "extras" / "python"
                    for directory in install_root.iterdir()
                    if directory.is_dir()
                ),
                reverse=True,
            )
        )

    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))
        try:
            import ncu_report as module

            return module
        except ImportError:
            continue
    return None


def _metric_scalar(action, name: str) -> float | None:
    try:
        metric = action.metric_by_name(name)
    except Exception:
        return None
    if not metric or not metric.has_value():
        return None
    try:
        value = float(metric.as_double())
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _all_scalar_metrics(action) -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        names = action.metric_names()
    except Exception:
        return values
    for name in names:
        value = _metric_scalar(action, name)
        if value is not None:
            values[name] = value
    return values


def _first_metric(values: dict[str, float], *names: str) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    return None


def _kernel_info(action, scalar_metrics: dict[str, float] | None = None) -> dict[str, Any]:
    values = scalar_metrics if scalar_metrics is not None else _all_scalar_metrics(action)
    duration_ns = _first_metric(values, "gpu__time_duration.sum")
    result: dict[str, Any] = {
        "registers_per_thread": int(values.get("launch__registers_per_thread", 0)),
        "registers_allocated": int(
            values.get("launch__registers_per_thread_allocated", 0)
        ),
        "block_size": int(values.get("launch__block_size", 0)),
        "grid_size": int(values.get("launch__grid_size", 0)),
        "shared_memory_bytes": int(
            values.get("launch__shared_mem_per_block_dynamic", 0)
            + values.get("launch__shared_mem_per_block_static", 0)
        ),
    }
    if duration_ns is not None:
        result["duration_us"] = duration_ns / 1000.0

    aliases = {
        "sm_throughput_pct": (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        "memory_throughput_pct": (
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        "achieved_occupancy_pct": (
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ),
        "eligible_warps_per_scheduler": (
            "smsp__warps_eligible.avg.per_cycle_active",
        ),
        "active_warps_per_scheduler": (
            "smsp__warps_active.avg.per_cycle_active",
        ),
    }
    for label, names in aliases.items():
        value = _first_metric(values, *names)
        if value is not None:
            result[label] = value
    return result


def _parse_dest_regs(sass: str) -> set[int]:
    value = re.sub(r"^@[!]?P\d+\s+", "", sass)
    for prefix, width in _WIDE_WRITE_OPS.items():
        if value.startswith(prefix):
            if width == 0:
                return set()
            match = _REG_PATTERN.search(value[len(prefix.split(".")[0]) :])
            if not match:
                return set()
            base = int(match.group(1))
            return set(range(base, base + width))

    opcode = value.split()[0] if value else ""
    if opcode in {"EXIT", "BRA", "NOP", "BAR", "MEMBAR", "RET"}:
        return set()
    if "SETP" in opcode:
        return set()
    match = _REG_PATTERN.search(value[len(opcode) :])
    return {int(match.group(1))} if match else set()


def _parse_source_regs(sass: str) -> set[int]:
    value = re.sub(r"^@[!]?P\d+\s+", "", sass)
    opcode = value.split()[0] if value else ""
    if "STG" in opcode:
        return {int(register) for register in _REG_PATTERN.findall(value)}
    operands = [item.strip() for item in value[len(opcode) :].strip().split(",")]
    sources: set[int] = set()
    for index, operand in enumerate(operands):
        if index:
            sources.update(int(register) for register in _REG_PATTERN.findall(operand))
    return sources


def _opcode(sass: str) -> str:
    value = re.sub(r"^@[!]?P\d+\s+", "", sass)
    full = value.split()[0] if value else ""
    return full.split(".")[0]


def _compute_live_registers(instructions: list[_SassInstr]) -> None:
    definitions = {item.pc: _parse_dest_regs(item.sass) for item in instructions}
    uses = {item.pc: _parse_source_regs(item.sass) for item in instructions}
    first_definition: dict[int, int] = {}
    last_use: dict[int, int] = {}
    for index, instruction in enumerate(instructions):
        for register in definitions[instruction.pc]:
            first_definition.setdefault(register, index)
        for register in uses[instruction.pc]:
            last_use[register] = index
    for register, first in first_definition.items():
        last_use[register] = max(last_use.get(register, first), first)
    for index, instruction in enumerate(instructions):
        instruction.live_regs = sum(
            first <= index <= last_use.get(register, -1)
            for register, first in first_definition.items()
        )


def _attribute_stalls(instructions: list[_SassInstr]) -> None:
    instruction_map = {item.pc: item for item in instructions}
    last_writer: dict[int, int] = {}
    writers_before: dict[int, dict[int, int]] = {}
    for instruction in instructions:
        writers_before[instruction.pc] = dict(last_writer)
        for register in _parse_dest_regs(instruction.sass):
            last_writer[register] = instruction.pc

    attributed: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for instruction in instructions:
        for reason, samples in instruction.stalls.items():
            if reason in _NON_STALL_REASONS:
                continue
            if reason not in _BACKWARD_ATTRIBUTE_REASONS:
                attributed[instruction.pc][reason] += samples
                continue
            target_ops = (
                _LONG_LATENCY_OPS if reason == "long_scoreboard" else _MIO_OPS
            )
            producers = {
                writers_before.get(instruction.pc, {}).get(register)
                for register in _parse_source_regs(instruction.sass)
            }
            producers = {
                pc
                for pc in producers
                if pc in instruction_map and _opcode(instruction_map[pc].sass) in target_ops
            }
            if producers:
                for pc in producers:
                    attributed[pc][reason] += samples / len(producers)
            else:
                attributed[instruction.pc][reason] += samples
    for instruction in instructions:
        instruction.stalls = dict(attributed.get(instruction.pc, {}))


def _collect_instructions(action) -> OrderedDict[int, _SassInstr]:
    instructions: OrderedDict[int, _SassInstr] = OrderedDict()
    try:
        metric = action.metric_by_name("inst_executed")
    except Exception:
        return instructions
    if not metric or not metric.has_correlation_ids():
        return instructions
    correlations = metric.correlation_ids()
    for index in range(metric.num_instances()):
        pc = correlations.as_uint64(index)
        try:
            source_info = action.source_info(pc)
        except Exception:
            source_info = None
        try:
            sass = action.sass_by_pc(pc).strip()
        except Exception:
            sass = ""
        try:
            ptx = action.ptx_by_pc(pc).strip()
        except Exception:
            ptx = ""
        instructions[pc] = _SassInstr(
            pc=pc,
            sass=sass,
            ptx=ptx,
            file=source_info.file_name() if source_info else "",
            line=source_info.line() if source_info else -1,
            exec_count=metric.as_double(index),
        )
    return instructions


def _collect_stalls(action, instructions: OrderedDict[int, _SassInstr]) -> None:
    try:
        metric_names = action.metric_names()
    except Exception:
        return
    for name in metric_names:
        if not name.startswith(_STALL_PREFIX) or "_not_issued" in name:
            continue
        try:
            metric = action.metric_by_name(name)
        except Exception:
            continue
        if not metric or not metric.has_correlation_ids() or metric.num_instances() == 0:
            continue
        reason = name[len(_STALL_PREFIX) :]
        correlations = metric.correlation_ids()
        for index in range(metric.num_instances()):
            pc = correlations.as_uint64(index)
            value = metric.as_double(index)
            if value > 0 and pc in instructions:
                instructions[pc].stalls[reason] = (
                    instructions[pc].stalls.get(reason, 0.0) + value
                )


def _collect_source_markers(action, ncu_report_module):
    line_markers: dict[tuple[str, int], list[str]] = defaultdict(list)
    pc_markers: dict[int, list[str]] = defaultdict(list)
    if not hasattr(action, "source_markers"):
        return line_markers, pc_markers
    source_kind = getattr(ncu_report_module, "MarkerKind_SOURCE", None)
    sass_kind = getattr(ncu_report_module, "MarkerKind_SASS", None)
    for marker in action.source_markers():
        message = marker.get("message", "")
        kind = marker.get("kind", -1)
        if kind == source_kind:
            location = marker.get("source_location", {})
            filename = location.get("file_name", "")
            line = location.get("line", -1)
            if filename and line >= 0:
                line_markers[(filename, line)].append(message)
        elif kind == sass_kind:
            pc = marker.get("source_address", 0)
            if pc:
                pc_markers[pc].append(message)
    return line_markers, pc_markers


def _short_pc(pc: int) -> str:
    return f"0x{pc & 0xFFFFFF:06x}"


def _top_reasons(stalls: dict[str, float], limit: int = 2) -> str:
    total = sum(stalls.values())
    if total <= 0:
        return ""
    parts = []
    for reason, value in sorted(stalls.items(), key=lambda item: -item[1])[:limit]:
        percentage = value / total * 100
        if percentage >= 5:
            parts.append(f"{reason} {percentage:.0f}%")
    return ", ".join(parts)


def _format_instruction_kernel(action, ncu_report_module):
    instructions_by_pc = _collect_instructions(action)
    data: dict[str, Any] = {
        "name": action.name(),
        "info": _kernel_info(action),
        "sass": [],
        "source_summary": [],
    }
    if not instructions_by_pc:
        return "  No instruction-level data found.\n", data, 0, 0

    _collect_stalls(action, instructions_by_pc)
    line_markers, pc_markers = _collect_source_markers(action, ncu_report_module)
    instructions = sorted(instructions_by_pc.values(), key=lambda item: item.pc)
    _compute_live_registers(instructions)
    _attribute_stalls(instructions)
    total_stalls = sum(sum(item.stalls.values()) for item in instructions)

    source_content: dict[str, dict[int, str]] = {}
    try:
        source_files = action.source_files()
    except Exception:
        source_files = {}
    for filename, content in source_files.items():
        lines = content.splitlines()
        source_content[filename] = {
            index + 1: text for index, text in enumerate(lines)
        }

    by_line: dict[tuple[str, int], list[_SassInstr]] = defaultdict(list)
    mapped = 0
    for instruction in instructions:
        attributed = (
            sum(instruction.stalls.values()) / total_stalls * 100
            if total_stalls
            else 0.0
        )
        entry: dict[str, Any] = {
            "pc": _short_pc(instruction.pc),
            "sass": instruction.sass,
            "file": Path(instruction.file).name if instruction.file else "",
            "line": instruction.line,
            "exec": int(instruction.exec_count),
            "live_regs": instruction.live_regs,
            "stall_pct": round(attributed, 2),
            "top_reasons": _top_reasons(instruction.stalls),
        }
        if pc_markers.get(instruction.pc):
            entry["markers"] = pc_markers[instruction.pc]
        data["sass"].append(entry)
        if instruction.file and instruction.line >= 0:
            mapped += 1
            by_line[(instruction.file, instruction.line)].append(instruction)

    output = io.StringIO()
    info = data["info"]
    print(f"  Kernel: {action.name()}", file=output)
    print(
        "  Instructions: "
        f"{len(instructions)} ({mapped} source-mapped), "
        f"registers={info['registers_per_thread']}/thread, "
        f"block={info['block_size']}, grid={info['grid_size']}",
        file=output,
    )
    print("  Source Summary", file=output)
    for filename, line in sorted(by_line):
        line_instructions = by_line[(filename, line)]
        stalls: dict[str, float] = defaultdict(float)
        executions = 0.0
        for instruction in line_instructions:
            executions += instruction.exec_count
            for reason, value in instruction.stalls.items():
                stalls[reason] += value
        percentage = sum(stalls.values()) / total_stalls * 100 if total_stalls else 0.0
        code = source_content.get(filename, {}).get(line, "").strip()
        pcs = [_short_pc(item.pc) for item in line_instructions]
        summary: dict[str, Any] = {
            "file": Path(filename).name,
            "line": line,
            "code": code,
            "stall_pct": round(percentage, 2),
            "top_reasons": _top_reasons(stalls),
            "exec": int(executions),
            "pcs": pcs,
        }
        if line_markers.get((filename, line)):
            summary["markers"] = line_markers[(filename, line)]
        data["source_summary"].append(summary)
        print(f"    Line {line}: {code}", file=output)
        print(
            f"      PCs: {', '.join(pcs)}; exec={int(executions):,}; "
            f"stall={percentage:.2f}% {_top_reasons(stalls)}",
            file=output,
        )
    if not by_line:
        print("    No source mappings were embedded in the report.", file=output)
    return output.getvalue(), data, len(instructions), mapped


def _format_metrics_summary(metrics: dict[str, Any]) -> str:
    output = io.StringIO()
    print("NCU Metrics Summary", file=output)
    print("===================", file=output)
    for kernel in metrics["kernels"]:
        summary = kernel["summary"]
        print(
            f"[{kernel['invocation']}] {kernel['name']}",
            file=output,
        )
        labels = (
            ("duration_us", "duration"),
            ("sm_throughput_pct", "SM throughput"),
            ("memory_throughput_pct", "memory throughput"),
            ("achieved_occupancy_pct", "achieved occupancy"),
            ("eligible_warps_per_scheduler", "eligible warps/scheduler"),
        )
        for key, label in labels:
            if key in summary:
                suffix = " us" if key == "duration_us" else ""
                print(f"  {label}: {summary[key]:.4f}{suffix}", file=output)
        print(
            f"  grid={summary['grid_size']}, block={summary['block_size']}, "
            f"registers={summary['registers_per_thread']}/thread, "
            f"shared_memory={summary['shared_memory_bytes']} bytes",
            file=output,
        )
    return output.getvalue()


def analyze_ncu_report(
    report_path: str | Path,
    *,
    include_instructions: bool,
) -> NcuAnalysis:
    """Return normalized metrics and optional source-mapped SASS evidence."""

    module = _import_ncu_report()
    if module is None:
        raise NcuReportError(
            "ncu_report Python module was not found; cannot normalize the NCU report"
        )
    try:
        report = module.load_report(str(report_path))
    except Exception as exc:
        raise NcuReportError(f"could not load NCU report {report_path}: {exc}") from exc

    kernels: list[dict[str, Any]] = []
    actions = []
    invocation = 0
    for run in report:
        for action in run:
            invocation += 1
            scalar_metrics = _all_scalar_metrics(action)
            kernels.append(
                {
                    "name": action.name(),
                    "invocation": invocation,
                    "summary": _kernel_info(action, scalar_metrics),
                    "raw_metrics": scalar_metrics,
                }
            )
            actions.append(action)
    if not kernels:
        raise NcuReportError("NCU report did not contain any kernel actions")

    metrics = {"kernel_invocation_count": len(kernels), "kernels": kernels}
    text = _format_metrics_summary(metrics)
    sass: dict[str, Any] = {"kernels": []}
    instruction_count = 0
    mapped_instruction_count = 0

    if include_instructions:
        text += "\nNCU Instruction Summary\n=======================\n"
        seen: set[str] = set()
        for action in actions:
            name = action.name()
            if name in seen:
                continue
            seen.add(name)
            kernel_text, kernel_data, count, mapped = _format_instruction_kernel(
                action, module
            )
            text += kernel_text
            sass["kernels"].append(kernel_data)
            instruction_count += count
            mapped_instruction_count += mapped

    return NcuAnalysis(
        text=text,
        metrics=metrics,
        sass=sass,
        instruction_count=instruction_count,
        mapped_instruction_count=mapped_instruction_count,
    )
