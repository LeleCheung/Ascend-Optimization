# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Structured parsing and rendering for Iluvatar ixsys traces."""

from __future__ import annotations

import ast
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# Older ixsys reports may retain this optional legacy range. Native profiling
# now uses cudaProfilerStart/Stop as its authoritative boundary, so absence of
# the range is diagnostic only.
MEASURED_RANGE_NAME = "kernelgen_profile"

_KERNEL_COLUMNS = (
    "ts",
    "dur",
    "name",
    "deviceId",
    "contextId",
    "streamId",
    "correlationId",
    "grid",
    "block",
    "regs",
    "srfs",
    "staticSharedMemory",
    "dynamicSharedMemory",
    "latency",
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dimensions(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [_as_int(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, (list, tuple)):
        return [_as_int(item) for item in parsed]
    return []


def _relation_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _round_us(value: float) -> float:
    return round(value, 6)


def parse_ixsys_trace(path: Path, *, iterations: int) -> Dict[str, Any]:
    """Parse measured kernel launches from an ixsys SQLite trace.

    The runner opens the CUDA profiler window only around measured candidate
    calls. Consequently every row in cuda_kernel belongs to that window; no
    launch-count heuristic is needed for input generation or warmup.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("ixsys trace is missing or empty")

    uri = f"file:{path.resolve()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        if not _relation_exists(connection, "cuda_kernel"):
            raise ValueError("ixsys trace has no cuda_kernel relation")
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(cuda_kernel)")
        }
        missing = [column for column in _KERNEL_COLUMNS if column not in columns]
        if missing:
            raise ValueError(
                "ixsys cuda_kernel schema is missing columns: " + ", ".join(missing)
            )
        rows = list(
            connection.execute(
                "SELECT " + ", ".join(_KERNEL_COLUMNS) + " FROM cuda_kernel ORDER BY ts"
            )
        )
        measured_range_observed = False
        if _relation_exists(connection, "cuda_nvtx"):
            measured_range_observed = (
                connection.execute(
                    "SELECT 1 FROM cuda_nvtx WHERE name = ? LIMIT 1",
                    (MEASURED_RANGE_NAME,),
                ).fetchone()
                is not None
            )
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"could not read ixsys trace: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()

    if not rows:
        raise ValueError("ixsys trace contains no measured CUDA kernel launches")

    warnings: List[str] = []
    if not measured_range_observed:
        warnings.append(
            f"ixsys trace did not retain the {MEASURED_RANGE_NAME!r} NVTX range"
        )

    launches: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for sequence, row in enumerate(rows):
        duration_ns = _as_int(row["dur"], -1)
        if duration_ns < 0:
            warnings.append(f"ignored kernel launch {sequence} with negative duration")
            continue
        name = str(row["name"] or "<unnamed>")
        grid = _dimensions(row["grid"])
        block = _dimensions(row["block"])
        launch = {
            "sequence": sequence,
            "timestamp_ns": _as_int(row["ts"]),
            "duration_us": _round_us(duration_ns / 1000.0),
            "op_name": name,
            "device_id": _as_int(row["deviceId"]),
            "context_id": _as_int(row["contextId"]),
            "stream_id": _as_int(row["streamId"]),
            "correlation_id": _as_int(row["correlationId"]),
            "grid": grid,
            "block": block,
            "vector_registers_per_thread": _as_int(row["regs"]),
            "scalar_registers_per_warp": _as_int(row["srfs"]),
            "static_shared_memory_bytes": _as_int(row["staticSharedMemory"]),
            "dynamic_shared_memory_bytes": _as_int(row["dynamicSharedMemory"]),
            "launch_latency_us": _round_us(max(_as_int(row["latency"]), 0) / 1000.0),
        }
        launches.append(launch)
        key = (
            name,
            tuple(grid),
            tuple(block),
            launch["vector_registers_per_thread"],
            launch["scalar_registers_per_warp"],
            launch["static_shared_memory_bytes"],
            launch["dynamic_shared_memory_bytes"],
        )
        grouped[key].append(launch)

    if not launches:
        raise ValueError("ixsys trace contains no valid measured CUDA kernel launches")

    operations: List[Dict[str, Any]] = []
    for key, op_launches in grouped.items():
        durations = [item["duration_us"] for item in op_launches]
        latencies = [item["launch_latency_us"] for item in op_launches]
        total_duration = sum(durations)
        operation = {
            "op_name": key[0],
            "grid": list(key[1]),
            "block": list(key[2]),
            "vector_registers_per_thread": key[3],
            "scalar_registers_per_warp": key[4],
            "static_shared_memory_bytes": key[5],
            "dynamic_shared_memory_bytes": key[6],
            "measured_invocations": len(op_launches),
            "launches_per_iteration": round(len(op_launches) / iterations, 6),
            "total_duration_us": _round_us(total_duration),
            "avg_duration_us": _round_us(total_duration / len(op_launches)),
            "avg_contribution_us": _round_us(total_duration / iterations),
            "min_duration_us": _round_us(min(durations)),
            "max_duration_us": _round_us(max(durations)),
            "avg_launch_latency_us": _round_us(sum(latencies) / len(latencies)),
            "min_launch_latency_us": _round_us(min(latencies)),
            "max_launch_latency_us": _round_us(max(latencies)),
        }
        operations.append(operation)

    operations.sort(
        key=lambda item: (item["avg_contribution_us"], item["op_name"]),
        reverse=True,
    )
    total_kernel_time = sum(item["duration_us"] for item in launches)
    first_timestamp = min(item["timestamp_ns"] for item in launches)
    last_timestamp = max(
        item["timestamp_ns"] + round(item["duration_us"] * 1000) for item in launches
    )
    avg_time_us = _round_us(total_kernel_time / iterations)
    summary = {
        "avg_time_us": avg_time_us,
        "operation_count": len(operations),
        "kernel_invocation_count": len(launches),
        # Retained for compatibility with pre-normalization clients.
        "kernel_launch_count": len(launches),
        "top_operations": operations[:10],
    }
    metrics = {
        "avg_time_us": avg_time_us,
        "total_kernel_time_us": _round_us(total_kernel_time),
        "measured_iterations": iterations,
        "kernel_invocation_count": len(launches),
        # Retained for compatibility with pre-normalization clients.
        "kernel_launch_count": len(launches),
        "measured_kernel_span_us": _round_us(
            (last_timestamp - first_timestamp) / 1000.0
        ),
        "measured_range": MEASURED_RANGE_NAME,
        "measured_range_observed": measured_range_observed,
        "ops": operations,
    }
    return {
        "summary": summary,
        "metrics": metrics,
        "kernel_launches": launches,
        "warnings": warnings,
    }


def _format_dimensions(value: Sequence[int]) -> str:
    return "x".join(str(item) for item in value) if value else ""


def render_ixsys_details(
    parsed: Mapping[str, Any],
    *,
    device: str,
    warmup: int,
    iterations: int,
) -> str:
    """Render factual, agent-readable ixsys kernel metrics."""
    summary = parsed["summary"]
    metrics = parsed["metrics"]
    lines = [
        "Iluvatar ixsys Metrics Summary",
        "==============================",
        "",
        "Section: Overview",
        "",
        f"Device: {device}",
        f"Warmup Invocations: {warmup}",
        f"Measured Invocations: {iterations}",
        "Collection Window: cudaProfilerStart/Stop around measured calls only",
        "Timing Authority: diagnostic only; evaluator latency is authoritative",
        f"Average Kernel Time Per Candidate Call (us): {summary['avg_time_us']:.6f}",
        f"Total Measured Kernel Time (us): {metrics['total_kernel_time_us']:.6f}",
        f"Kernel Invocations: {summary['kernel_invocation_count']}",
        f"Distinct Kernel Operations: {summary['operation_count']}",
        "",
        "Section: Kernel Summary",
        "",
        "Kernel | Grid | Block | Calls | Calls/Candidate | Avg Launch(us) | "
        "Avg Contribution(us) | Min(us) | Max(us) | VRF | SRF | Shared(bytes)",
    ]
    for operation in metrics["ops"]:
        shared = (
            operation["static_shared_memory_bytes"]
            + operation["dynamic_shared_memory_bytes"]
        )
        lines.append(
            f"{operation['op_name']} | "
            f"{_format_dimensions(operation['grid'])} | "
            f"{_format_dimensions(operation['block'])} | "
            f"{operation['measured_invocations']} | "
            f"{operation['launches_per_iteration']:.6f} | "
            f"{operation['avg_duration_us']:.6f} | "
            f"{operation['avg_contribution_us']:.6f} | "
            f"{operation['min_duration_us']:.6f} | "
            f"{operation['max_duration_us']:.6f} | "
            f"{operation['vector_registers_per_thread']} | "
            f"{operation['scalar_registers_per_warp']} | {shared}"
        )
    lines.extend(
        [
            "",
            "Section: Performance Signals",
            "",
            f"Measured NVTX Range Observed: {metrics['measured_range_observed']}",
            f"Measured Kernel Span (us): {metrics['measured_kernel_span_us']:.6f}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_ixsys_structured_artifacts(
    parsed: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.write_text(
        json.dumps(parsed["metrics"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fieldnames = [
        "op_name",
        "grid",
        "block",
        "measured_invocations",
        "launches_per_iteration",
        "total_duration_us",
        "avg_duration_us",
        "avg_contribution_us",
        "min_duration_us",
        "max_duration_us",
        "avg_launch_latency_us",
        "vector_registers_per_thread",
        "scalar_registers_per_warp",
        "static_shared_memory_bytes",
        "dynamic_shared_memory_bytes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for operation in parsed["metrics"]["ops"]:
            row = {key: operation.get(key, "") for key in fieldnames}
            row["grid"] = _format_dimensions(operation["grid"])
            row["block"] = _format_dimensions(operation["block"])
            writer.writerow(row)
