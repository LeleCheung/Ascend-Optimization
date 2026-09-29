"""Cambricon CNPerf and Triton-MLU profiling."""

from __future__ import annotations

import csv
import json
import os
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
from ..source_view import report_source_target
from .cambricon_report import (
    _IR_FORMATS,
    _render_instruction_report,
    _render_instruction_summary,
    _render_source_report,
    _source_mapping_payload,
    parse_cnperf_timeline,
    parse_ir_source_operations,
    parse_kernel_statistics,
    parse_mlisa,
    parse_pmu_csv,
)


def _tool_available(executable: str) -> bool:
    path = Path(executable)
    return path.is_file() if path.is_absolute() else shutil.which(executable) is not None


class CnperfProfiler(ManagedProfiler):
    backend = "mlu"
    name = "cnperf-cli+triton-mlu"

    def _default_executable(self) -> str:
        return os.environ.get("KGS_CNPERF_PATH", "cnperf-cli")

    def available(self) -> bool:
        return _tool_available(self._default_executable())

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("mlu", {}).get(
            "cnperf_path", self._default_executable()
        )
        return _tool_available(str(configured))

    def levels(self) -> List[str]:
        levels = ["metrics"]
        if _tool_available("cnas"):
            levels.append("instruction")
        return levels

    def capabilities(self) -> List[str]:
        return [
            "kernel_profile", "performance_counters", "execution_timeline",
            "device_binary", "compiler_ir", "source_mapping", "instruction_listing",
        ]

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("mlu", {})
        allowed = {"cnperf_path", "pmu", "events", "replay_mode"}
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ProfilerExecutionError(
                f"unsupported Cambricon profiler options: {', '.join(unknown)}"
            )
        executable = str(options.get("cnperf_path", self._default_executable()))
        if not _tool_available(executable):
            raise ProfilerUnsupportedError(
                f"cnperf-cli executable was not found at {executable!r}"
            )
        if context.options.level == "instruction" and not _tool_available("cnas"):
            raise ProfilerUnsupportedError(
                "cnas is required for Cambricon instruction profiling"
            )
        pmu = options.get("pmu", True)
        if not isinstance(pmu, bool):
            raise ProfilerExecutionError("mlu.pmu must be a boolean")
        replay = str(options.get("replay_mode", "none"))
        if replay not in {"none", "kernel"}:
            raise ProfilerExecutionError("mlu.replay_mode must be 'none' or 'kernel'")
        events_value = options.get("events", "")
        if isinstance(events_value, list):
            events = ",".join(str(value) for value in events_value if value)
        elif events_value is None or isinstance(events_value, str):
            events = events_value or ""
        else:
            raise ProfilerExecutionError("mlu.events must be a string or list")
        cache = context.artifact_dir / "triton-cache"
        cache.mkdir(exist_ok=False)
        return {
            "executable": executable,
            "pmu": pmu,
            "events": events,
            "replay": replay,
            "cache": cache,
            "report": context.artifact_dir / "profile.cnperf-rep",
        }

    def _run_export(
        self,
        context: ProfileExecutionContext,
        command: list[str],
        label: str,
        *,
        required: bool,
    ) -> bool:
        execution = context.run_tool(command, label=label)
        log = context.artifact_dir / f"{label.replace(' ', '-')}.log"
        log.write_text(execution.output, encoding="utf-8")
        context.add_artifact("profile_log", "text", log, "text/plain")
        if execution.returncode != 0:
            message = f"{label} exited with status {execution.returncode}"
            if required:
                raise ProfilerExecutionError(message)
            context.warnings.append(message)
            return False
        return True

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        command = [
            prepared["executable"], "record", "-c", "0",
            "--capture_range", "cnProfilerApi", "--device_task", "kernel",
            "--device_task_mode", "detail", "--force_overwrite",
            "-o", str(prepared["report"]),
        ]
        if prepared["pmu"]:
            command.append("--pmu")
        if prepared["events"]:
            command.extend(["--events", prepared["events"]])
        if prepared["replay"] == "kernel":
            command.extend(["--replay_mode", "kernel"])
        command.extend(context.command.argv)
        overrides = {
            "TRITON_CACHE_DIR": str(prepared["cache"]),
            "KGS_CACHE_PATH": str(context.artifact_dir / "module-cache"),
        }
        execution = context.run_profile_stage(
            command, label="CNPerf collection", env_overrides=overrides
        )
        log = context.artifact_dir / "cnperf-record.log"
        log.write_text(execution.output, encoding="utf-8")
        context.add_artifact(
            "profile_log", "text", log, "text/plain; charset=utf-8"
        )
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"cnperf-cli exited with status {execution.returncode}"
            )
        report = prepared["report"]
        if not report.is_file() or report.stat().st_size == 0:
            raise ProfilerExecutionError(
                "cnperf-cli completed without producing a non-empty .cnperf-rep report"
            )
        context.add_artifact("vendor_report", "cnperf-rep", report)
        context.warnings.extend(
            [
                "CNPerf timing is diagnostic and is not evaluator latency",
                "the normalized timeline keeps only kernels inside cnProfilerStart/Stop",
            ]
        )
        if prepared["replay"] == "kernel":
            context.warnings.append(
                "CNPerf kernel replay may re-execute stateful kernels"
            )

        report_dir = context.artifact_dir / "cnperf-report-csv"
        kernel_dir = context.artifact_dir / "cnperf-kernel-csv"
        timeline_dir = context.artifact_dir / "cnperf-timeline"
        for path in (report_dir, kernel_dir, timeline_dir):
            path.mkdir(exist_ok=False)
        report_ok = self._run_export(
            context,
            [prepared["executable"], "report", "--csv", "--force_overwrite",
             "-o", str(report_dir), str(report)],
            "cnperf-report-export",
            required=False,
        )
        kernel_ok = self._run_export(
            context,
            [prepared["executable"], "kernel", "--csv", "--force_overwrite",
             "-o", str(kernel_dir), str(report)],
            "cnperf-kernel-export",
            required=False,
        )
        self._run_export(
            context,
            [prepared["executable"], "timechart", "--detail", "-o",
             str(timeline_dir), "--name", "trace", str(report)],
            "cnperf-timechart-export",
            required=True,
        )
        timeline = timeline_dir / "trace.json"
        if not timeline.is_file() or timeline.stat().st_size == 0:
            raise ProfilerExecutionError("CNPerf did not produce an execution timeline")
        context.add_artifact(
            "execution_timeline", "chrome-trace-json", timeline, "application/json"
        )
        return {
            "timeline": timeline,
            "report_dir": report_dir,
            "kernel_dir": kernel_dir,
            "report_ok": report_ok,
            "kernel_ok": kernel_ok,
        }

    def _instruction_artifacts(
        self, context: ProfileExecutionContext, cache: Path
    ) -> dict[str, Any]:
        target = report_source_target(context.source_roots)
        instructions: list[dict[str, Any]] = []
        source_operations: list[dict[str, Any]] = []
        binaries: set[Path] = set()
        mlisa_paths = sorted(cache.rglob("*.mlisa"))
        for mlisa in mlisa_paths:
            text = mlisa.read_text(encoding="utf-8", errors="replace")
            parsed = parse_mlisa(text, binary_name=mlisa.name)
            instructions.extend(parsed)
            context.add_artifact("isa_dump", "mlisa", mlisa, "text/plain")
            binary = mlisa.with_suffix(".cnbin")
            if binary.is_file():
                binaries.add(binary)
                context.add_artifact("device_binary", "cnbin", binary)
            kernel_hint = parsed[0]["kernel"] if parsed else mlisa.stem
            for suffix, format_name in _IR_FORMATS.items():
                ir = mlisa.with_suffix(suffix)
                if not ir.is_file():
                    continue
                context.add_artifact("compiler_ir", format_name, ir, "text/plain")
                if suffix == ".ttir":
                    source_operations.extend(
                        parse_ir_source_operations(
                            ir.read_text(encoding="utf-8", errors="replace"),
                            ir_name=ir.name,
                            target=target,
                            kernel_hint=kernel_hint,
                        )
                    )
        source_mapping = _source_mapping_payload(source_operations, target)
        payload = {
            "evidence_type": "static_native_instruction_assembly",
            "instruction_stage": "mlisa",
            "native_machine_isa": True,
            "machine_isa_form": "compiler-emitted-mlisa",
            "post_link_binary_disassembly": False,
            "instruction_addresses": False,
            "instruction_count": len(instructions),
            "instructions": instructions,
        }
        listing = context.artifact_dir / "instructions.json"
        listing.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mapping = context.artifact_dir / "source-map.json"
        mapping.write_text(json.dumps(source_mapping, indent=2), encoding="utf-8")
        summary_text = _render_instruction_summary(
            instructions,
            source_mapping,
            mlisa_count=len(mlisa_paths),
            binary_count=len(binaries),
        )
        summary_path = context.artifact_dir / "instruction-summary.txt"
        summary_path.write_text(summary_text, encoding="utf-8")
        report = context.artifact_dir / "instruction-report.txt"
        report.write_text(_render_instruction_report(instructions), encoding="utf-8")
        source_report = context.artifact_dir / "source-report.txt"
        source_report.write_text(_render_source_report(source_mapping), encoding="utf-8")
        for kind, format_name, path, media in (
            ("instruction_listing", "json", listing, "application/json"),
            ("source_mapping", "json", mapping, "application/json"),
            ("instruction_summary", "text", summary_path, "text/plain"),
            ("instruction_report", "text", report, "text/plain"),
            ("source_report", "text", source_report, "text/plain"),
        ):
            context.add_artifact(kind, format_name, path, media)
        context.warnings.extend(
            [
                "MLISA is compiler-emitted pre-link native assembly without PC addresses",
                "source mappings are source-to-TTIR-operation, not per-instruction",
            ]
        )
        return {
            "mlisa_file_count": len(mlisa_paths),
            "binary_count": len(binaries),
            "instruction_count": len(instructions),
            "mapped_compiler_operation_count": source_mapping[
                "mapped_compiler_operation_count"
            ],
            "mapped_instruction_count": 0,
            "source_mapping_granularity": source_mapping["mapping_granularity"],
        }

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: dict[str, Any],
    ) -> None:
        try:
            metrics = parse_cnperf_timeline(collected["timeline"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ProfilerExecutionError(
                f"could not normalize CNPerf timeline: {exc}"
            ) from exc
        statistics = next(collected["report_dir"].rglob("KernelStatistics.csv"), None)
        if collected["report_ok"] and statistics:
            try:
                metrics["vendor_kernel_statistics"] = parse_kernel_statistics(statistics)
            except (OSError, csv.Error, ValueError) as exc:
                context.warnings.append(f"could not parse KernelStatistics.csv: {exc}")
        counters = []
        if collected["kernel_ok"]:
            for path in sorted(collected["kernel_dir"].rglob("*.csv")):
                context.add_artifact("performance_counters", "csv", path, "text/csv")
                try:
                    parsed = parse_pmu_csv(path)
                except (OSError, csv.Error, ValueError) as exc:
                    context.warnings.append(f"could not parse {path.name}: {exc}")
                    continue
                if parsed:
                    counters.append(parsed)
        metrics["performance_counters"] = counters
        metrics["counter_unit_count"] = len(counters)
        metrics["counter_metric_count"] = sum(item["counter_count"] for item in counters)
        metrics["measured_iterations"] = context.options.iterations
        context.metrics = metrics
        context.summary.update(
            {
                "kernel_count": metrics["kernel_count"],
                "kernel_invocation_count": metrics["kernel_invocation_count"],
                "total_kernel_time_us": metrics["total_kernel_time_us"],
                "counter_unit_count": metrics["counter_unit_count"],
                "counter_metric_count": metrics["counter_metric_count"],
            }
        )
        context.add_capability("kernel_profile", "execution_timeline")
        if counters:
            context.add_capability("performance_counters")
        if context.options.level == "instruction":
            context.summary.update(self._instruction_artifacts(context, prepared["cache"]))
            context.add_capability("instruction_listing")
            for artifact_kind in ("device_binary", "compiler_ir"):
                if context.has_artifact(artifact_kind):
                    context.add_capability(artifact_kind)
            if context.summary.get("mapped_compiler_operation_count", 0):
                context.add_capability("source_mapping")
            else:
                context.warnings.append(
                    "Cambricon TTIR could not be mapped back to submitted source"
                )
        metrics_path = context.artifact_dir / "cnperf-metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        context.add_artifact(
            "normalized_metrics", "json", metrics_path, "application/json"
        )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError(
                "CNPerf timeline contains no captured device kernels"
            )
        if context.options.level == "instruction" and context.summary.get(
            "instruction_count", 0
        ) < 1:
            raise ProfilerExecutionError(
                "Triton-MLU cache did not yield an MLISA instruction listing"
            )


__all__ = ["CnperfProfiler"]
