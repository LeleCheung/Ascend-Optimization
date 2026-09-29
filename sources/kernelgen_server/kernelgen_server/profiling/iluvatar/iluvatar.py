"""Iluvatar ixsys and IXKN profiling for evaluator-owned commands."""

from __future__ import annotations

import json
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
from .iluvatar_ixkn import (
    build_ixkn_instruction_listing,
    merge_ixkn_instruction_listings,
    merge_ixkn_instruction_reports,
    merge_ixkn_source_mappings,
    parse_ixkn_assembly_listing,
    parse_ixkn_details_csv,
    parse_ixkn_source_mapping,
    render_ixkn_instruction_details,
    render_ixkn_instruction_summary,
    render_ixkn_source_details,
    write_ixkn_instruction_artifacts,
    write_ixkn_source_mapping,
)
from .iluvatar_metrics import (
    parse_ixsys_trace,
    render_ixsys_details,
    write_ixsys_structured_artifacts,
)


_ALLOWED_OPTIONS = {
    "ixsys_path",
    "ixkn_path",
    "ixsys_traces",
    "telemetry",
    "gpu_metrics",
    "gpu_metrics_set",
    "gpu_metrics_frequency",
    "kernel_name",
    "ixkn_max_kernels",
    "set",
    "triton_disable_line_info",
    "llvm_extract_di_local_variables",
}


class IxknProfiler(ManagedProfiler):
    backend = "iluvatar"
    name = "ixsys+ixkn"

    @staticmethod
    def _ixsys_available(path: str = "ixsys") -> bool:
        return shutil.which(path) is not None

    @staticmethod
    def _ixkn_available(path: str = "ixkn-cli") -> bool:
        return shutil.which(path) is not None

    def available(self) -> bool:
        return self._ixsys_available()

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("iluvatar", {})
        return self._ixsys_available(str(configured.get("ixsys_path", "ixsys")))

    def capabilities(self) -> List[str]:
        return [
            "kernel_profile",
            "execution_timeline",
            "source_mapping",
            "compiler_ir",
            "instruction_listing",
        ]

    def normalize_options(
        self, options: ProfileOptions
    ) -> tuple[ProfileOptions, List[str]]:
        backend_options = dict(options.backend_options)
        iluvatar_options = dict(backend_options.get("iluvatar", {}))
        if "ixkn_max_kernels" not in iluvatar_options:
            return options, []
        maximum, warning = self._normalize_max_kernels(
            iluvatar_options["ixkn_max_kernels"]
        )
        iluvatar_options["ixkn_max_kernels"] = maximum
        backend_options["iluvatar"] = iluvatar_options
        return (
            options.model_copy(update={"backend_options": backend_options}),
            [warning] if warning else [],
        )

    def levels(self) -> List[str]:
        levels = ["metrics"] if self._ixsys_available() else []
        if self._ixsys_available() and self._ixkn_available():
            levels.append("instruction")
        return levels

    def levels_for(self, options: ProfileOptions) -> List[str]:
        configured = options.backend_options.get("iluvatar", {})
        ixsys = str(configured.get("ixsys_path", "ixsys"))
        ixkn = str(configured.get("ixkn_path", "ixkn-cli"))
        levels = ["metrics"] if self._ixsys_available(ixsys) else []
        if levels and self._ixkn_available(ixkn):
            levels.append("instruction")
        return levels

    @staticmethod
    def _normalize_max_kernels(value: Any) -> tuple[int, str | None]:
        default = 10
        try:
            if isinstance(value, bool):
                raise ValueError
            normalized = int(value)
        except (TypeError, ValueError, OverflowError):
            return default, f"invalid iluvatar.ixkn_max_kernels={value!r}; using {default}"
        if normalized < 1:
            return 1, "iluvatar.ixkn_max_kernels was clamped to 1"
        if normalized > 10:
            return 10, "iluvatar.ixkn_max_kernels was clamped to 10"
        return normalized, None

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("iluvatar", {})
        unknown = sorted(set(options) - _ALLOWED_OPTIONS)
        if unknown:
            raise ProfilerExecutionError(
                "unsupported Iluvatar profiler options: " + ", ".join(unknown)
            )
        ixsys = str(options.get("ixsys_path", "ixsys"))
        if not self._ixsys_available(ixsys):
            raise ProfilerUnsupportedError(
                f"ixsys executable was not found at {ixsys!r}"
            )
        ixkn = str(options.get("ixkn_path", "ixkn-cli"))
        if context.options.level == "instruction" and not self._ixkn_available(ixkn):
            raise ProfilerUnsupportedError(
                f"ixkn-cli executable was not found at {ixkn!r}"
            )
        cache = context.artifact_dir / "triton-cache"
        cache.mkdir(exist_ok=False)
        return {
            "options": options,
            "ixsys": ixsys,
            "ixkn": ixkn,
            "trace": context.artifact_dir / "ixsys-trace.sqlite",
            "cache": cache,
        }

    @staticmethod
    def _trace_names(options: dict[str, Any]) -> list[str]:
        value = options.get("ixsys_traces", ["cuda_api", "cuda_kernel"])
        if isinstance(value, str):
            names = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, list):
            names = [str(item).strip() for item in value if str(item).strip()]
        else:
            raise ProfilerExecutionError(
                "iluvatar.ixsys_traces must be a list or comma-separated string"
            )
        if "cuda_kernel" not in names:
            names.append("cuda_kernel")
        if options.get("telemetry"):
            for name in ("mem", "power", "temperature"):
                if name not in names:
                    names.append(name)
        return names

    def _collect_metrics(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        options = prepared["options"]
        physical_device = context.command.env.get("CUDA_VISIBLE_DEVICES", "0")
        command = [
            prepared["ixsys"],
            "--output",
            str(prepared["trace"]),
            "--devices",
            physical_device,
            "--trace",
            ",".join(self._trace_names(options)),
            "--profile-from-start",
            "off",
        ]
        if options.get("gpu_metrics"):
            command.extend(
                [
                    "--gpu-metrics-device",
                    physical_device,
                    "--gpu-metrics-set",
                    str(options.get("gpu_metrics_set", 2)),
                    "--gpu-metrics-frequency",
                    str(options.get("gpu_metrics_frequency", 500)),
                ]
            )
        command.extend(context.command.argv)
        execution = context.run_profile_stage(
            command,
            label="Iluvatar ixsys metrics collection",
            env_overrides={
                "TRITON_CACHE_DIR": str(prepared["cache"]),
                "TRITON_DISABLE_LINE_INFO": (
                    "0" if context.options.level == "instruction" else "1"
                ),
                "LLVM_EXTRACT_DI_LOCAL_VARIABLES": str(
                    options.get("llvm_extract_di_local_variables", "1")
                ),
            },
        )
        log = context.artifact_dir / "ixsys.log"
        log.write_text(execution.output, encoding="utf-8")
        context.add_artifact("profile_log", "text", log, "text/plain")
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"ixsys exited with status {execution.returncode}"
            )
        if not prepared["trace"].is_file() or prepared["trace"].stat().st_size == 0:
            raise ProfilerExecutionError("ixsys did not produce a non-empty trace")
        context.add_artifact(
            "execution_timeline",
            "ixsys-sqlite",
            prepared["trace"],
            "application/vnd.sqlite3",
        )
        try:
            return parse_ixsys_trace(
                prepared["trace"], iterations=context.options.iterations
            )
        except ValueError as exc:
            raise ProfilerExecutionError(f"could not parse ixsys trace: {exc}") from exc

    def _source_paths(self, context: ProfileExecutionContext) -> dict[str, str]:
        result: dict[str, str] = {}
        for root in context.source_roots:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    result[str(path.resolve())] = path.relative_to(root).as_posix()
        return result

    def _export_ixkn_page(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        report: Path,
        argv: list[str],
        *,
        label: str,
        output_path: Path,
    ) -> str | None:
        execution = context.run_tool(
            [prepared["ixkn"], "-i", str(report), *argv],
            label=label,
        )
        output_path.write_text(execution.output, encoding="utf-8")
        context.add_artifact("vendor_export", "text", output_path, "text/plain")
        if execution.returncode != 0:
            context.warnings.append(
                f"{label} exited with status {execution.returncode}"
            )
            return None
        return execution.output

    def _collect_instruction(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        options = prepared["options"]
        configured = str(options.get("kernel_name", "")).strip()
        discovered = [
            str(item.get("op_name", "")).strip()
            for item in metrics.get("ops", [])
            if str(item.get("op_name", "")).strip()
        ]
        kernels = list(dict.fromkeys([configured] if configured else discovered))
        maximum, warning = self._normalize_max_kernels(
            options.get("ixkn_max_kernels", 10)
        )
        if warning:
            context.warnings.append(warning)
        kernels = kernels[:maximum]
        if not kernels:
            raise ProfilerExecutionError(
                "ixsys metrics did not discover a kernel for IXKN collection"
            )
        source_paths = self._source_paths(context)
        source_reports: list[Dict[str, Any]] = []
        listings: list[Dict[str, Any]] = []
        counter_reports: list[Dict[str, Any]] = []
        report_count = 0
        collected_kernels: list[str] = []
        failed_kernels: list[str] = []
        for index, kernel in enumerate(kernels):
            report = context.artifact_dir / f"ixkn-report-{index:03d}.ixkn-rep"
            command = [
                prepared["ixkn"],
                "--force-overwrite",
                "--set",
                str(options.get("set", "full")),
                "--devices",
                "0",
                "--kernel-name",
                kernel,
                "--launch-count",
                "1",
                "--profile-from-start",
                "off",
                "--import-source",
                "on",
            ]
            if source_paths:
                command.extend(
                    ["--resolve-source-file", ",".join(source_paths)]
                )
            command.extend(["-o", str(report), *context.command.argv])
            execution = context.run_profile_stage(
                command,
                label=f"Iluvatar IXKN collection {index}",
                env_overrides={
                    "TRITON_CACHE_DIR": str(prepared["cache"]),
                    "TRITON_DISABLE_LINE_INFO": str(
                        options.get("triton_disable_line_info", "0")
                    ),
                    "LLVM_EXTRACT_DI_LOCAL_VARIABLES": str(
                        options.get("llvm_extract_di_local_variables", "1")
                    ),
                },
            )
            log = context.artifact_dir / f"ixkn-collect-{index:03d}.log"
            log.write_text(execution.output, encoding="utf-8")
            context.add_artifact("profile_log", "text", log, "text/plain")
            if (
                execution.returncode != 0
                or not report.is_file()
                or report.stat().st_size == 0
            ):
                context.warnings.append(
                    f"IXKN did not produce a complete report for {kernel!r}"
                )
                failed_kernels.append(kernel)
                continue
            report_count += 1
            context.add_artifact("vendor_report", "ixkn-rep", report)
            listing_count_before = len(listings)

            source_text = self._export_ixkn_page(
                context,
                prepared,
                report,
                ["--page", "source", "--print-source", "cuda,assembly"],
                label=f"Iluvatar IXKN source export {index}",
                output_path=context.artifact_dir / f"ixkn-source-{index:03d}.txt",
            )
            source_listing = None
            if source_text is not None:
                try:
                    source_report = parse_ixkn_source_mapping(
                        source_text,
                        source_paths=source_paths,
                        expected_kernel=kernel,
                    )
                    source_reports.append(source_report)
                    source_listing = build_ixkn_instruction_listing(source_report)
                    if source_listing.get("instruction_count", 0):
                        listings.append(source_listing)
                except ValueError as exc:
                    context.warnings.append(
                        f"IXKN source mapping unavailable for {kernel!r}: {exc}"
                    )

            if source_listing is None or not source_listing.get("instruction_count", 0):
                assembly_text = self._export_ixkn_page(
                    context,
                    prepared,
                    report,
                    ["--page", "source", "--print-source", "assembly"],
                    label=f"Iluvatar IXKN assembly export {index}",
                    output_path=context.artifact_dir / f"ixkn-assembly-{index:03d}.txt",
                )
                if assembly_text is not None:
                    try:
                        listings.append(
                            parse_ixkn_assembly_listing(
                                assembly_text, expected_kernel=kernel
                            )
                        )
                        context.warnings.append(
                            f"IXKN source mapping unavailable for {kernel!r}; "
                            "using assembly-only instruction listing"
                        )
                    except ValueError as exc:
                        context.warnings.append(
                            f"IXKN assembly could not be parsed for {kernel!r}: {exc}"
                        )

            if len(listings) > listing_count_before:
                collected_kernels.append(kernel)
            else:
                failed_kernels.append(kernel)
                context.warnings.append(
                    f"IXKN produced no instruction listing for {kernel!r}"
                )

            ir_text = self._export_ixkn_page(
                context,
                prepared,
                report,
                ["--page", "source", "--print-source", "IR"],
                label=f"Iluvatar IXKN IR export {index}",
                output_path=context.artifact_dir / f"ixkn-ir-{index:03d}.txt",
            )
            if ir_text:
                context.add_artifact(
                    "compiler_ir",
                    "text",
                    context.artifact_dir / f"ixkn-ir-{index:03d}.txt",
                    "text/plain",
                )
                context.add_capability("compiler_ir")

            details = context.run_tool(
                [
                    prepared["ixkn"],
                    "-i",
                    str(report),
                    "--page",
                    "details",
                    "--csv",
                ],
                label=f"Iluvatar IXKN details export {index}",
            )
            if details.returncode == 0:
                try:
                    parsed, warnings = parse_ixkn_details_csv(details.output)
                    counter_reports.append(parsed)
                    context.warnings.extend(warnings)
                except ValueError as exc:
                    context.warnings.append(
                        f"IXKN details could not be parsed for {kernel!r}: {exc}"
                    )

        result: dict[str, Any] = {
            "selected_kernel_count": len(kernels),
            "selected_kernels": kernels,
            "collected_kernel_count": len(collected_kernels),
            "collected_kernels": collected_kernels,
            "failed_kernel_count": len(failed_kernels),
            "failed_kernels": failed_kernels,
            "ixkn_report_count": report_count,
        }
        merged_listing = None
        if listings:
            merged_listing = merge_ixkn_instruction_listings(listings)
            path = context.artifact_dir / "instructions.json"
            path.write_text(json.dumps(merged_listing, indent=2), encoding="utf-8")
            context.add_artifact(
                "instruction_listing", "json", path, "application/json"
            )
            context.add_capability("instruction_listing")
            result.update(
                {
                    "instruction_count": merged_listing["instruction_count"],
                    "mapped_instruction_count": merged_listing[
                        "mapped_instruction_count"
                    ],
                }
            )
        if source_reports:
            merged_source = merge_ixkn_source_mappings(source_reports)
            mapping = context.artifact_dir / "source-mapping.json"
            details = context.artifact_dir / "source-details.txt"
            write_ixkn_source_mapping(merged_source, mapping)
            details.write_text(
                render_ixkn_source_details(merged_source), encoding="utf-8"
            )
            context.add_artifact(
                "source_mapping", "ixkn-source-json", mapping, "application/json"
            )
            context.add_artifact("source_details", "text", details, "text/plain")
            context.add_capability("source_mapping")
        counter_report = None
        if counter_reports:
            counter_report = merge_ixkn_instruction_reports(counter_reports)
            json_path = context.artifact_dir / "ixkn-counters.json"
            csv_path = context.artifact_dir / "ixkn-counters.csv"
            details_path = context.artifact_dir / "ixkn-counter-details.txt"
            write_ixkn_instruction_artifacts(
                counter_report, json_path=json_path, csv_path=csv_path
            )
            details_path.write_text(
                render_ixkn_instruction_details(counter_report), encoding="utf-8"
            )
            context.add_artifact(
                "instruction_counters", "json", json_path, "application/json"
            )
            context.add_artifact("instruction_counters", "csv", csv_path, "text/csv")
            context.add_artifact(
                "instruction_counters", "text", details_path, "text/plain"
            )
        if merged_listing is not None:
            summary_path = context.artifact_dir / "instruction-summary.txt"
            summary_path.write_text(
                render_ixkn_instruction_summary(merged_listing, counter_report),
                encoding="utf-8",
            )
            context.add_artifact(
                "instruction_summary", "text", summary_path, "text/plain"
            )
        return result

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        parsed = self._collect_metrics(context, prepared)
        instruction = {}
        if context.options.level == "instruction":
            instruction = self._collect_instruction(
                context, prepared, parsed["metrics"]
            )
        return {"parsed": parsed, "instruction": instruction}

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: dict[str, Any],
    ) -> None:
        del prepared
        parsed = collected["parsed"]
        json_path = context.artifact_dir / "ixsys-metrics.json"
        csv_path = context.artifact_dir / "kernel-summary.csv"
        details_path = context.artifact_dir / "profile-details.txt"
        write_ixsys_structured_artifacts(
            parsed, json_path=json_path, csv_path=csv_path
        )
        details_path.write_text(
            render_ixsys_details(
                parsed,
                device=context.device,
                warmup=context.options.warmup,
                iterations=context.options.iterations,
            ),
            encoding="utf-8",
        )
        context.add_artifact(
            "normalized_metrics", "json", json_path, "application/json"
        )
        context.add_artifact("kernel_summary", "csv", csv_path, "text/csv")
        context.add_artifact("profile_details", "text", details_path, "text/plain")
        context.metrics = dict(parsed["metrics"])
        context.summary.update(parsed["summary"])
        context.summary.update(collected["instruction"])
        context.warnings.extend(parsed["warnings"])
        context.warnings.append(
            "ixsys device-kernel timing is diagnostic and is not evaluator latency"
        )
        context.add_capability("kernel_profile", "execution_timeline")

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError(
                "ixsys trace contains no device kernel invocation"
            )
        if (
            context.options.level == "instruction"
            and context.summary.get("instruction_count", 0) < 1
        ):
            raise ProfilerExecutionError(
                "IXKN produced neither source-mapped nor assembly instructions"
            )


__all__ = ["IxknProfiler"]
