"""Kunlunxin XProfiler profiling for evaluator-owned commands."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..base import (
    ManagedProfiler,
    ProfileExecutionContext,
    ProfilerExecutionError,
    ProfilerUnsupportedError,
)
from ..models import ProfileOptions
from .kunlunxin_report import _parse_xprofiler_report, _render_xprofiler_details, _round


_XPROFILER_ENV = "KGS_XPROFILER_PATH"
_MEASUREMENT_MARKERS = {
    "start": re.compile(r"^KGS_PROFILE_START_NS=(\d+)$", re.MULTILINE),
    "end": re.compile(r"^KGS_PROFILE_END_NS=(\d+)$", re.MULTILINE),
    "wall": re.compile(
        r"^KGS_PROFILE_WALL_TIME_US=([0-9]+(?:\.[0-9]+)?)$", re.MULTILINE
    ),
}


def _resolve_xprofiler(configured: object = None) -> str | None:
    for candidate in (configured, os.environ.get(_XPROFILER_ENV), "xprofiler"):
        if candidate is None or not str(candidate).strip():
            continue
        if resolved := shutil.which(str(candidate).strip()):
            return str(Path(resolved).resolve())
    return None


def _measurement_window(stdout: str) -> tuple[int, int, float]:
    values: Dict[str, str] = {}
    for name, pattern in _MEASUREMENT_MARKERS.items():
        matches = pattern.findall(stdout)
        if len(matches) != 1:
            raise ProfilerExecutionError(
                f"xprofiler runner did not report exactly one {name} measurement marker"
            )
        values[name] = matches[0]
    start_ns, end_ns = int(values["start"]), int(values["end"])
    wall_time_us = float(values["wall"])
    if end_ns <= start_ns or wall_time_us <= 0:
        raise ProfilerExecutionError(
            "xprofiler runner reported an invalid measurement window"
        )
    return start_ns, end_ns, wall_time_us


def _traced_status(stdout: str) -> int | None:
    marker = "trace exited, status="
    index = stdout.rfind(marker)
    if index == -1:
        return None
    match = re.match(r"-?\d+", stdout[index + len(marker) :].strip())
    return int(match.group()) if match else None


class XProfilerProfiler(ManagedProfiler):
    backend = "kunlunxin"
    name = "xprofiler"

    def available(self) -> bool:
        return _resolve_xprofiler() is not None

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("kunlunxin", {}).get(
            "xprofiler_path"
        )
        return _resolve_xprofiler(configured) is not None

    def capabilities(self) -> List[str]:
        return ["kernel_profile", "execution_timeline"]

    def levels(self) -> List[str]:
        return ["metrics"]

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("kunlunxin", {})
        allowed = {"xprofiler_path", "buffer_mb", "resolution", "ld_library_path"}
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ProfilerExecutionError(
                f"unsupported Kunlunxin profiler options: {', '.join(unknown)}"
            )
        executable = _resolve_xprofiler(options.get("xprofiler_path"))
        if executable is None:
            requested = options.get("xprofiler_path") or os.environ.get(
                _XPROFILER_ENV, "xprofiler"
            )
            raise ProfilerUnsupportedError(
                f"xprofiler executable was not found at {str(requested)!r}"
            )
        return {
            "options": options,
            "executable": executable,
            "report": context.artifact_dir / "xprofiler-trace.json",
        }

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> str:
        options = prepared["options"]
        command = [prepared["executable"], "--xpu=0"]
        try:
            buffer_mb = int(options.get("buffer_mb", 512))
        except (TypeError, ValueError) as exc:
            raise ProfilerExecutionError("kunlunxin.buffer_mb must be an integer") from exc
        if buffer_mb > 0:
            command.extend(["-b", str(buffer_mb)])
        if options.get("resolution") is not None:
            try:
                command.extend(["-r", str(int(options["resolution"]))])
            except (TypeError, ValueError) as exc:
                raise ProfilerExecutionError(
                    "kunlunxin.resolution must be an integer"
                ) from exc
        command.extend(["-s", "-e", str(prepared["report"]), *context.command.argv])
        overrides = {"CUDA_DEVICE_ORDER": "OAM_ID"}
        if extra := options.get("ld_library_path"):
            existing = context.environment().get("LD_LIBRARY_PATH", "")
            overrides["LD_LIBRARY_PATH"] = (
                f"{extra}:{existing}" if existing else str(extra)
            )
        execution = context.run_profile_stage(
            command,
            label="Kunlunxin XProfiler collection",
            env_overrides=overrides,
            cwd=context.artifact_dir,
        )
        log = context.artifact_dir / "xprofiler-log.txt"
        log.write_text(execution.output, encoding="utf-8")
        context.add_artifact(
            "profile_log", "text", log, "text/plain; charset=utf-8"
        )
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"xprofiler exited with status {execution.returncode}"
            )
        traced_status = _traced_status(execution.output)
        if traced_status is None:
            raise ProfilerExecutionError(
                "xprofiler did not report the profiled application status"
            )
        if traced_status != 0:
            raise ProfilerExecutionError(
                f"the profiled application exited with status {traced_status}"
            )
        report = prepared["report"]
        if not report.is_file() or report.stat().st_size == 0:
            raise ProfilerExecutionError(
                "xprofiler did not produce a non-empty profile report"
            )
        context.add_artifact(
            "execution_timeline", "trace-json", report, "application/json"
        )
        return execution.output

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: str,
    ) -> None:
        start_ns, end_ns, wall_time_us = _measurement_window(collected)
        try:
            metrics = _parse_xprofiler_report(
                prepared["report"], start_ns=start_ns, end_ns=end_ns
            )
        except ValueError as exc:
            raise ProfilerExecutionError(str(exc)) from exc
        complete = "dma data out of space" not in collected
        metrics["complete"] = complete
        operations = metrics["ops"]
        avg_kernel = _round(metrics["kernel_time_us"] / context.options.iterations)
        metrics.update(
            {
                "avg_time_us": avg_kernel,
                "avg_kernel_time_us": avg_kernel,
                "avg_wall_time_us": _round(wall_time_us / context.options.iterations),
                "wall_time_us": _round(wall_time_us),
                "measured_iterations": context.options.iterations,
                "kernel_invocation_count": len(metrics["kernels"]),
                "operation_count": len(operations),
            }
        )
        summary = {
            "avg_time_us": avg_kernel,
            "avg_kernel_time_us": avg_kernel,
            "avg_wall_time_us": metrics["avg_wall_time_us"],
            "wall_time_us": metrics["wall_time_us"],
            "kernel_time_us": metrics["kernel_time_us"],
            "operation_count": len(operations),
            "kernel_invocation_count": len(metrics["kernels"]),
            "top_operations": operations[:10],
            "trace_complete": complete,
        }
        metrics_path = context.artifact_dir / "xprofiler-metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        context.add_artifact(
            "normalized_metrics", "json", metrics_path, "application/json"
        )
        details = _render_xprofiler_details(
            metrics,
            summary,
            device=context.device,
            warmup=context.options.warmup,
            iterations=context.options.iterations,
        )
        details_path = context.artifact_dir / "profile-details.txt"
        details_path.write_text(details, encoding="utf-8")
        context.add_artifact(
            "profile_details", "text", details_path, "text/plain; charset=utf-8"
        )
        context.metrics = metrics
        context.summary = {**summary, "text": details[:16_000]}
        context.add_capability("kernel_profile", "execution_timeline")
        context.warnings.extend(
            [
                "xprofiler tracing adds instrumentation overhead",
                "XProfiler device timing is diagnostic and is not eval latency",
            ]
        )
        if not complete:
            context.warnings.append(
                "xprofiler dropped trace samples; increase kunlunxin.buffer_mb"
            )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError(
                "XProfiler trace contains no device kernel events"
            )


__all__ = ["XProfilerProfiler", "_measurement_window", "_traced_status"]
