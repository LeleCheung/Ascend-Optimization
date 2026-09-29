# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Nsight Compute profiling for evaluator-owned commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, List

from ..base import (
    ManagedProfiler,
    ProfileExecutionContext,
    ProfilerExecutionError,
    ProfilerUnsupportedError,
)
from ..models import ProfileOptions
from .nvidia_report import NcuReportError, _import_ncu_report, analyze_ncu_report


_MAX_NCU_PROFILE_ITERATIONS = 1


class NcuProfiler(ManagedProfiler):
    backend = "cuda"
    name = "ncu"

    def available(self) -> bool:
        return shutil.which("ncu") is not None and _import_ncu_report() is not None

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("cuda", {}).get("ncu_path", "ncu")
        return shutil.which(str(configured)) is not None and _import_ncu_report() is not None

    def capabilities(self) -> List[str]:
        return ["performance_counters", "kernel_profile", "instruction_listing"]

    def levels(self) -> List[str]:
        return ["metrics", "instruction"]

    def normalize_options(
        self, options: ProfileOptions
    ) -> tuple[ProfileOptions, List[str]]:
        if options.iterations <= _MAX_NCU_PROFILE_ITERATIONS:
            return options, []
        effective = options.model_copy(
            update={"iterations": _MAX_NCU_PROFILE_ITERATIONS}
        )
        return effective, [
            f"cuda profile iterations clamped from {options.iterations} to "
            f"{_MAX_NCU_PROFILE_ITERATIONS}; NCU may internally replay each kernel "
            "to collect performance counters"
        ]

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        backend_options = context.options.backend_options.get("cuda", {})
        allowed = {"ncu_path", "set"}
        unknown = sorted(set(backend_options) - allowed)
        if unknown:
            raise ProfilerExecutionError(
                f"unsupported cuda profiler options: {', '.join(unknown)}"
            )
        executable = str(backend_options.get("ncu_path", "ncu"))
        if shutil.which(executable) is None:
            raise ProfilerUnsupportedError(
                f"NCU executable was not found at {executable!r}"
            )
        if _import_ncu_report() is None:
            raise ProfilerUnsupportedError(
                "ncu_report Python module was not found; cannot normalize NCU reports"
            )
        report_base = context.artifact_dir / "report"
        report_path = context.artifact_dir / "report.ncu-rep"
        ncu_set = backend_options.get(
            "set", "detailed" if context.options.level == "metrics" else "full"
        )
        if not isinstance(ncu_set, str) or not ncu_set.strip():
            raise ProfilerExecutionError("cuda.set must be a non-empty string")
        return {
            "executable": executable,
            "report_base": report_base,
            "report_path": report_path,
            "set": ncu_set,
        }

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> Any:
        command = [
            prepared["executable"],
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--force-overwrite",
            "--page",
            "details",
            "--set",
            prepared["set"],
            "--import-source",
            "on",
        ]
        if context.source_roots:
            command.extend(
                ["--source-folders", ";".join(str(path) for path in context.source_roots)]
            )
        command.extend(
            ["--export", str(prepared["report_base"]), *context.command.argv]
        )
        execution = context.run_profile_stage(command, label="NCU collection")
        raw_log = context.artifact_dir / "ncu-raw.txt"
        raw_log.write_text(execution.output, encoding="utf-8")
        context.add_artifact(
            "profile_details", "text", raw_log, "text/plain; charset=utf-8"
        )
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"ncu exited with status {execution.returncode}"
            )
        report_path = prepared["report_path"]
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise ProfilerExecutionError(
                "ncu completed without producing a non-empty report.ncu-rep"
            )
        context.add_artifact("vendor_report", "ncu-rep", report_path)
        return report_path

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: Path,
    ) -> None:
        del prepared
        try:
            analysis = analyze_ncu_report(
                collected,
                include_instructions=context.options.level == "instruction",
            )
        except NcuReportError as exc:
            raise ProfilerExecutionError(str(exc)) from exc

        metrics_path = context.artifact_dir / "ncu-metrics.json"
        metrics_path.write_text(
            json.dumps(analysis.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        context.add_artifact(
            "normalized_metrics", "json", metrics_path, "application/json"
        )
        context.metrics = analysis.metrics
        context.summary.update(
            {
                "text": analysis.text[:16_000],
                "kernel_invocation_count": analysis.metrics.get(
                    "kernel_invocation_count", 0
                ),
            }
        )
        context.add_capability("performance_counters", "kernel_profile")
        context.warnings.append(
            "NCU instrumented timing is diagnostic and is not eval latency"
        )

        if context.options.level == "instruction":
            instruction_summary = context.artifact_dir / "instruction-summary.txt"
            instruction_summary.write_text(analysis.text, encoding="utf-8")
            sass_path = context.artifact_dir / "sass.json"
            sass_path.write_text(
                json.dumps(analysis.sass, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            context.add_artifact(
                "instruction_summary",
                "text",
                instruction_summary,
                "text/plain; charset=utf-8",
            )
            context.add_artifact(
                "instruction_listing", "json", sass_path, "application/json"
            )
            context.summary.update(
                {
                    "instruction_count": analysis.instruction_count,
                    "mapped_instruction_count": analysis.mapped_instruction_count,
                }
            )
            context.add_capability("instruction_listing")
            if analysis.mapped_instruction_count:
                context.add_capability("source_hotspots")
            else:
                context.warnings.append(
                    "NCU did not embed source mappings for the instruction listing"
                )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError(
                "NCU report did not contain any kernel actions"
            )
        if (
            context.options.level == "instruction"
            and context.summary.get("instruction_count", 0) < 1
        ):
            raise ProfilerExecutionError(
                "NCU report did not contain an instruction listing"
            )


__all__ = ["NcuProfiler"]
