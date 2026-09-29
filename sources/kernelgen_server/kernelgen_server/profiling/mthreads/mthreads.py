"""Moore Threads MCU profiling for evaluator-owned commands."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, List

from ..base import (
    ManagedProfiler,
    ProfileExecutionContext,
    ProfilerExecutionError,
    ProfilerUnsupportedError,
)
from ..models import ProfileOptions
from ..resource import host_global_lock
from .mthreads_report import McuReportError, analyze_mcu_output


_MCU_LOCK_PATH = Path("/tmp/kernelgen-mcu-global.lock")
_MCU_ARGUMENTS = (
    ("dump_mode", "--dump-mode"),
    ("replay_mode", "--replay-mode"),
    ("kernel_name", "--kernel-name"),
    ("kernel_name_base", "--kernel-name-base"),
    ("sampling_interval", "--sampling-interval"),
    ("sampling_buffer_size", "--sampling-buffer-size"),
)


class McuProfiler(ManagedProfiler):
    backend = "musa"
    name = "mcu"

    def available(self) -> bool:
        return shutil.which(os.environ.get("KGS_MCU_PATH", "mcu")) is not None

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("musa", {}).get(
            "mcu_path", os.environ.get("KGS_MCU_PATH", "mcu")
        )
        return shutil.which(str(configured)) is not None

    def capabilities(self) -> List[str]:
        return ["performance_counters"]

    def levels(self) -> List[str]:
        return ["metrics"]

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("musa", {})
        allowed = {"mcu_path", *(name for name, _ in _MCU_ARGUMENTS)}
        unsupported = sorted(set(options) - allowed)
        if unsupported:
            raise ProfilerExecutionError(
                f"unsupported MCU options: {', '.join(unsupported)}"
            )
        configured = str(
            options.get("mcu_path") or os.environ.get("KGS_MCU_PATH") or "mcu"
        )
        executable = shutil.which(configured)
        if executable is None:
            raise ProfilerUnsupportedError(
                f"MCU executable was not found at {configured!r}"
            )
        return {"options": options, "executable": executable}

    def resource_scope(
        self, context: ProfileExecutionContext, prepared: Any
    ) -> AbstractContextManager[Any]:
        @contextmanager
        def locked():
            with host_global_lock(_MCU_LOCK_PATH, context.deadline) as waited:
                prepared["lock_wait_seconds"] = round(waited, 6)
                yield

        return locked()

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> str:
        report_base = context.artifact_dir / "report"
        command = [
            prepared["executable"], "-f", "-o", str(report_base),
            "--page", "details",
        ]
        for name, flag in _MCU_ARGUMENTS:
            value = prepared["options"].get(name)
            if value not in (None, ""):
                command.extend([flag, str(value)])
        command.extend(context.command.argv)
        execution = context.run_profile_stage(command, label="MCU collection")
        raw_log = context.artifact_dir / "mcu-raw.txt"
        raw_log.write_text(execution.output, encoding="utf-8")
        context.add_artifact(
            "profile_log", "text", raw_log, "text/plain; charset=utf-8"
        )
        settings = {
            "profiler": self.name,
            "level": context.options.level,
            "executable": prepared["executable"],
            "assigned_device": context.device,
            "global_lock_path": str(_MCU_LOCK_PATH),
            "global_lock_wait_seconds": prepared.get("lock_wait_seconds", 0.0),
            "command": command,
            "warmup": context.options.warmup,
            "iterations": context.options.iterations,
        }
        settings_path = context.artifact_dir / "mcu-settings.json"
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        context.add_artifact(
            "profile_settings", "json", settings_path, "application/json"
        )
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"mcu exited with status {execution.returncode}"
            )
        if "==ERROR==" in execution.output:
            raise ProfilerExecutionError("the profiled application returned an error code")
        if "No kernels were profiled" in execution.output:
            raise ProfilerExecutionError("mcu profiled no kernels")
        report_path = context.artifact_dir / "report.mcu-rep"
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise ProfilerExecutionError(
                "mcu completed without producing a non-empty report.mcu-rep"
            )
        context.add_artifact("vendor_report", "mcu-rep", report_path)
        return execution.output

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: str,
    ) -> None:
        del prepared
        try:
            analysis = analyze_mcu_output(collected)
        except McuReportError as exc:
            raise ProfilerExecutionError(str(exc)) from exc
        metrics_path = context.artifact_dir / "mcu-metrics.json"
        metrics_path.write_text(
            json.dumps(analysis.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        context.add_artifact(
            "normalized_metrics", "json", metrics_path, "application/json"
        )
        details_path = context.artifact_dir / "profile-details.txt"
        details = json.dumps(analysis.metrics, ensure_ascii=False, indent=2)
        details_path.write_text(details, encoding="utf-8")
        context.add_artifact(
            "profile_details", "text", details_path, "text/plain; charset=utf-8"
        )
        context.metrics = analysis.metrics
        context.summary.update(
            {
                "text": details[:16_000],
                "kernel_invocation_count": analysis.metrics.get(
                    "kernel_invocation_count", 0
                ),
                "unique_kernel_count": analysis.metrics.get("unique_kernel_count", 0),
                "application_replay_passes": analysis.metrics.get(
                    "application_replay_passes", 0
                ),
            }
        )
        context.add_capability("performance_counters")
        context.warnings.append(
            "MCU instrumented timing is diagnostic and is not eval latency"
        )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError(
                "MCU details contained no parseable kernel metrics"
            )


__all__ = ["McuProfiler"]
