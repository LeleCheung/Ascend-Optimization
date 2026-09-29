"""Hygon DCU profiling for evaluator-owned commands."""

from __future__ import annotations

import json
import os
import re
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
from ..process import ProfileStageError
from ..source_view import report_source_target
from .dcu_report import (
    build_hygon_source_mapping,
    parse_hipprof_kernel_csv,
    parse_hipprof_pmc_csv,
    parse_hipprof_timeline,
    parse_hygon_isa,
    parse_rocprof_csv,
    render_hygon_instruction_summary,
    select_measured_hipprof_pmc_records,
    validate_timeline,
)
from .hygon_trace_compat import (
    HygonTraceCompatError,
    prepare_hygon_rocprof,
    rocprof_trace_libraries_available,
)


_USAGE_MARKERS = ("Usage:\n", "- this help.", "tracing options:")
_RUNNER_ERROR_MARKERS = (
    "Traceback (most recent call last):",
    "ModuleNotFoundError:",
    "ImportError:",
)
_TRACE_ERROR_MARKERS = (
    "failed to load",
    "cannot be preloaded",
    "libroctracer_tool.so",
    "libroctracer64.so",
)
_ALLOWED_OPTIONS = {
    "profiler_path",
    "llvm_objdump_path",
    "objdump_path",
    "stats_only",
    "disable_trace_compat",
    "trace_compat_cache",
    "pmc_type",
    "kernel_name",
    "kernel_stack",
    "hiptx_trace",
    "omp_trace",
    "rccl_trace",
    "extra_args",
}


def _looks_like_usage_text(text: str) -> bool:
    return sum(marker in text for marker in _USAGE_MARKERS) >= 2


class HipprofProfiler(ManagedProfiler):
    backend = "hygon"
    name = "hipprof/rocprof"

    def available(self) -> bool:
        return bool(self._detect_profiler_tool({})[0])

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("hygon", {})
        return bool(self._detect_profiler_tool(configured)[0])

    def capabilities(self) -> List[str]:
        return [
            "kernel_profile",
            "execution_timeline",
            "performance_counters",
            "instruction_listing",
            "source_mapping",
        ]

    def levels(self) -> List[str]:
        return ["metrics", "instruction"]

    def levels_for(self, options: ProfileOptions) -> List[str]:
        configured = options.backend_options.get("hygon", {})
        if not self.available_for(options):
            return []
        levels = ["metrics"]
        if self._find_objdump(configured):
            levels.append("instruction")
        return levels

    @staticmethod
    def _validate_extra_args(value: Any) -> list[str]:
        values = value or []
        if not isinstance(values, list) or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise ProfilerExecutionError(
                "hygon.extra_args must contain only non-empty strings"
            )
        return values

    @staticmethod
    def _find_rocprof() -> str:
        if shutil.which("rocprof"):
            return "rocprof"
        for root_value in dict.fromkeys(
            value
            for value in (
                os.environ.get("DTKROOT"),
                os.environ.get("ROCM_PATH"),
                "/opt/dtk",
            )
            if value
        ):
            root = Path(root_value).expanduser()
            for relative in (
                Path("rocprofiler/bin/rocprof"),
                Path("rocprofiler/rocprofiler/bin/rocprof"),
            ):
                candidate = root / relative
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
        return ""

    @staticmethod
    def _find_objdump(options: dict[str, Any]) -> str:
        configured = options.get("llvm_objdump_path") or options.get("objdump_path")
        if configured:
            resolved = shutil.which(str(configured))
            candidate = Path(resolved or str(configured)).expanduser()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
        resolved = shutil.which("llvm-objdump")
        if resolved:
            return str(Path(resolved).resolve())
        for root_value in dict.fromkeys(
            value
            for value in (
                os.environ.get("DTKROOT"),
                os.environ.get("ROCM_PATH"),
                "/opt/dtk",
                "/opt/dtk-25.04",
                "/opt/dtk-26.04",
            )
            if value
        ):
            root = Path(str(root_value)).expanduser()
            for relative in (
                Path("llvm/bin/llvm-objdump"),
                Path("aillvm/bin/llvm-objdump"),
                Path("bin/llvm-objdump"),
            ):
                candidate = root / relative
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
        return ""

    def _detect_profiler_tool(
        self, options: dict[str, Any]
    ) -> tuple[str, str]:
        configured = options.get("profiler_path")
        if configured:
            resolved = shutil.which(str(configured))
            candidate = Path(resolved or str(configured)).expanduser()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate), candidate.name
            return "", ""
        if shutil.which("hipprof"):
            return "hipprof", "hipprof"
        rocprof = self._find_rocprof()
        return (rocprof, "rocprof") if rocprof else ("", "")

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("hygon", {})
        unknown = sorted(set(options) - _ALLOWED_OPTIONS)
        if unknown:
            raise ProfilerExecutionError(
                "unsupported Hygon profiler options: " + ", ".join(unknown)
            )
        executable, tool = self._detect_profiler_tool(options)
        if not executable:
            raise ProfilerUnsupportedError(
                "neither hipprof nor rocprof was found on the Hygon system"
            )
        objdump = self._find_objdump(options)
        if context.options.level == "instruction" and not objdump:
            raise ProfilerUnsupportedError(
                "DTK llvm-objdump was not found for Hygon instruction profiling"
            )
        if (
            tool == "rocprof"
            and context.options.level == "instruction"
            and not options.get("stats_only", False)
            and not options.get("disable_trace_compat", False)
            and not rocprof_trace_libraries_available(executable)
        ):
            def run_compat_command(
                command: list[str] | tuple[str, ...], label: str
            ) -> tuple[int, str]:
                result = context.run_tool(
                    command,
                    label=f"Hygon ROCtracer {label}",
                    cwd=context.artifact_dir,
                )
                return result.returncode, result.output

            try:
                compat = prepare_hygon_rocprof(
                    executable,
                    options.get("trace_compat_cache"),
                    command_runner=run_compat_command,
                    deadline=context.deadline,
                )
            except HygonTraceCompatError as exc:
                raise ProfilerUnsupportedError(
                    f"Hygon ROCtracer compatibility setup failed: {exc}"
                ) from exc
            executable = str(compat.launcher)
        output = context.artifact_dir / f"{tool}-output"
        output.mkdir(exist_ok=False)
        cache = context.artifact_dir / "triton-cache"
        if context.options.level == "instruction":
            cache.mkdir(exist_ok=False)
        return {
            "options": options,
            "executable": executable,
            "tool": tool,
            "objdump": objdump,
            "output": output,
            "cache": cache,
        }

    def _build_command(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> list[str]:
        options = prepared["options"]
        output = prepared["output"]
        if prepared["tool"] == "hipprof":
            command = [
                prepared["executable"],
                "-d",
                str(output),
                "-o",
                str(output / "results.csv"),
            ]
            command.append("--stats" if options.get("stats_only") else "--hip-trace")
            if context.options.level == "metrics":
                command.extend(
                    ["--pmc", "--pmc-type", str(options.get("pmc_type", 3))]
                )
                if options.get("kernel_name"):
                    command.extend(["--kernel-name", str(options["kernel_name"])])
            if options.get("kernel_stack"):
                command.append("--kernel-stack")
            for flag in ("hiptx_trace", "omp_trace", "rccl_trace"):
                if options.get(flag):
                    command.append("--" + flag.replace("_", "-"))
        else:
            command = [
                prepared["executable"],
                "-o",
                str(output / "results.csv"),
                "--timestamp",
                "on",
            ]
            if options.get("stats_only"):
                command.append("--stats")
            elif context.options.level == "instruction":
                command.append("--sys-trace")
        command.extend(self._validate_extra_args(options.get("extra_args")))
        command.extend(context.command.argv)
        return command

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        overrides = {}
        if context.options.level == "instruction":
            overrides = {
                "TRITON_CACHE_DIR": str(prepared["cache"]),
                "TRITON_DISABLE_LINE_INFO": "0",
                "TRITON_ALWAYS_COMPILE": "1",
            }
        try:
            execution = context.run_profile_stage(
                self._build_command(context, prepared),
                label=f"Hygon {prepared['tool']} collection",
                env_overrides=overrides,
            )
        except ProfileStageError as exc:
            self._save_collection_output(context, prepared, exc.output)
            raise
        self._save_collection_output(context, prepared, execution.output)
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"{prepared['tool']} exited with status {execution.returncode}"
            )
        if any(marker in execution.output for marker in _RUNNER_ERROR_MARKERS):
            raise ProfilerExecutionError(
                f"{prepared['tool']} completed but the evaluator command failed"
            )
        if _looks_like_usage_text(execution.output[:4_000]):
            raise ProfilerExecutionError(
                f"{prepared['tool']} printed usage text instead of profiling"
            )
        return self._collect_outputs(context, prepared)

    @staticmethod
    def _save_collection_output(
        context: ProfileExecutionContext, prepared: dict[str, Any], output: str
    ) -> None:
        log = context.artifact_dir / f"{prepared['tool']}.log"
        log.write_text(output, encoding="utf-8")
        context.add_artifact("profile_log", "text", log, "text/plain")
        if prepared["tool"] != "hipprof":
            return
        # DTK writes PMC CSVs into the evaluator cwd even when -d/-o select
        # another report directory. Move only this invocation's PID-scoped
        # files, never sweep a shared Gems/Catalog checkout for other jobs.
        for pid in set(re.findall(r"^HIP_PROF:process id '(\d+)'$", output, re.MULTILINE)):
            source = Path(context.command.cwd) / f"pmc_results_{pid}.csv"
            target = prepared["output"] / source.name
            if source.is_file() and not source.is_symlink() and source.resolve() != target.resolve():
                shutil.move(str(source), str(target))

    def _collect_outputs(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        timelines: list[Path] = []
        kernel_csvs: list[Path] = []
        pmc_csvs: list[Path] = []
        for path in sorted(prepared["output"].rglob("*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            suffix = path.suffix.lower()
            if suffix == ".json":
                timelines.append(path)
                context.add_artifact(
                    "execution_timeline", "trace-json", path, "application/json"
                )
            elif re.match(r"pmc_results_.*\.csv$", name):
                pmc_csvs.append(path)
                context.add_artifact(
                    "performance_counters", "csv", path, "text/csv"
                )
            elif name.endswith(".kernel.csv") or suffix == ".csv":
                kernel_csvs.append(path)
                context.add_artifact("op_summary", "csv", path, "text/csv")
            elif suffix in {".html", ".db", ".txt"}:
                context.add_artifact(
                    "vendor_report",
                    suffix.lstrip("."),
                    path,
                    "text/plain" if suffix == ".txt" else "application/octet-stream",
                )
        return {
            "timelines": timelines,
            "kernel_csvs": kernel_csvs,
            "pmc_csvs": pmc_csvs,
        }

    def _parse_metrics(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        errors: list[str] = []
        parsed = None
        if prepared["tool"] == "hipprof":
            for path in collected["timelines"]:
                try:
                    parsed = parse_hipprof_timeline(
                        path, context.options.warmup, context.options.iterations
                    )
                    context.add_capability("execution_timeline", "kernel_profile")
                    break
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
            if parsed is None:
                for path in collected["kernel_csvs"]:
                    try:
                        parsed = parse_hipprof_kernel_csv(
                            path, context.options.warmup, context.options.iterations
                        )
                        context.add_capability("kernel_profile")
                        break
                    except Exception as exc:
                        errors.append(f"{path.name}: {exc}")
        else:
            for path in collected["kernel_csvs"]:
                try:
                    parsed = parse_rocprof_csv(
                        path, context.options.warmup, context.options.iterations
                    )
                    context.add_capability("kernel_profile")
                    break
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
            for path in collected["timelines"]:
                try:
                    validate_timeline(path)
                    context.add_capability("execution_timeline")
                    if parsed is None:
                        parsed = parse_hipprof_timeline(
                            path,
                            context.options.warmup,
                            context.options.iterations,
                        )
                        context.add_capability("kernel_profile")
                    break
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
        if parsed is None:
            raise ProfilerExecutionError(
                f"{prepared['tool']} produced no valid kernel timing data: "
                + ("; ".join(errors) or "no report was produced")
            )
        raw_pmc = None
        for path in collected["pmc_csvs"]:
            try:
                candidate = parse_hipprof_pmc_csv(path)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            if raw_pmc is None:
                raw_pmc = {"records": [], "record_count": 0, "counter_names": []}
            raw_pmc["records"].extend(candidate["records"])
            raw_pmc["record_count"] += candidate["record_count"]
            for name in candidate["counter_names"]:
                if name not in raw_pmc["counter_names"]:
                    raw_pmc["counter_names"].append(name)
        if raw_pmc is not None:
            # The trace uses demangled C++ names; PMC uses linker symbols.
            # Keep the original symbol and match only exact demangled names.
            symbols = sorted({r["kernel_name"] for r in raw_pmc["records"] if r["kernel_name"].startswith("_Z")})
            demangler = shutil.which("c++filt")
            if symbols and demangler:
                execution = context.run_tool(
                    [demangler, *symbols], label="Hygon PMC symbol demangling"
                )
                names = execution.output.splitlines()
                if execution.returncode == 0 and len(names) == len(symbols):
                    mapping = dict(zip(symbols, names))
                    for record in raw_pmc["records"]:
                        if record["kernel_name"] in mapping:
                            record["demangled_name"] = mapping[record["kernel_name"]]
        pmc = (
            select_measured_hipprof_pmc_records(raw_pmc, parsed)
            if raw_pmc is not None
            else None
        )
        if pmc is not None:
            context.add_capability("performance_counters")
        elif prepared["tool"] == "hipprof" and context.options.level == "metrics":
            context.warnings.append(
                "hipprof exported no valid measured PMC records"
            )
        return parsed, pmc

    @staticmethod
    def _iter_code_objects(cache: Path) -> list[Path]:
        return sorted(
            path
            for path in cache.rglob("*")
            if path.is_file() and path.suffix.lower() in {".hsaco", ".co", ".elf"}
        )

    def _build_instruction_artifacts(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> None:
        target = report_source_target(context.source_roots)
        binaries = self._iter_code_objects(prepared["cache"])
        raw_chunks: list[str] = []
        instructions: list[dict[str, Any]] = []
        disassembled = 0
        for index, binary in enumerate(binaries):
            context.add_artifact("device_binary", "hygon-code-object", binary)
            execution = context.run_tool(
                [
                    prepared["objdump"],
                    "--disassemble",
                    "--line-numbers",
                    "--demangle",
                    str(binary),
                ],
                label=f"Hygon llvm-objdump {index}",
            )
            if execution.returncode != 0:
                context.warnings.append(
                    f"llvm-objdump failed for {binary.name}"
                )
                continue
            disassembled += 1
            raw_chunks.extend(
                [f"===== {binary.name} =====", execution.output.rstrip(), ""]
            )
            instructions.extend(
                parse_hygon_isa(execution.output, binary=binary, target=target)
            )
        raw = context.artifact_dir / "isa-dump.txt"
        raw.write_text("\n".join(raw_chunks).rstrip() + "\n", encoding="utf-8")
        context.add_artifact("isa_dump", "text", raw, "text/plain")
        mapping = build_hygon_source_mapping(instructions)
        listing_payload = {
            "schema_version": 1,
            "profiler": "dtk-llvm-objdump",
            "evidence_type": "static_instruction_listing",
            "dynamic_timeline": False,
            "instruction_count": len(instructions),
            "mapped_instruction_count": mapping["mapped_instruction_count"],
            "instructions": instructions,
        }
        listing = context.artifact_dir / "instructions.json"
        listing.write_text(json.dumps(listing_payload, indent=2), encoding="utf-8")
        context.add_artifact(
            "instruction_listing", "json", listing, "application/json"
        )
        summary = context.artifact_dir / "instruction-summary.txt"
        summary.write_text(
            render_hygon_instruction_summary(
                instructions,
                binary_count=len(binaries),
                disassembled_binary_count=disassembled,
            ),
            encoding="utf-8",
        )
        context.add_artifact("instruction_summary", "text", summary, "text/plain")
        if mapping["mapped_instruction_count"]:
            mapping_path = context.artifact_dir / "source-map.json"
            mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
            context.add_artifact(
                "source_mapping", "json", mapping_path, "application/json"
            )
            context.add_capability("source_mapping")
        elif instructions:
            context.warnings.append(
                "Hygon instruction listing contains no source mapping"
            )
        context.summary.update(
            {
                "binary_count": len(binaries),
                "disassembled_binary_count": disassembled,
                "instruction_count": len(instructions),
                "mapped_instruction_count": mapping["mapped_instruction_count"],
            }
        )
        if instructions:
            context.add_capability("instruction_listing")

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: dict[str, Any],
    ) -> None:
        parsed, pmc = self._parse_metrics(context, prepared, collected)
        invocation_count = sum(
            int(item.get("measured_invocations", 0)) for item in parsed["ops"]
        )
        metrics = dict(parsed)
        metrics.update(
            {
                "kernel_invocation_count": invocation_count,
                "operation_count": len(parsed["ops"]),
                "measured_iterations": context.options.iterations,
            }
        )
        if pmc is not None:
            metrics["performance_counters"] = pmc
        context.metrics = metrics
        context.summary.update(
            {
                "tool": prepared["tool"],
                "avg_time_us": parsed["avg_time_us"],
                "operation_count": len(parsed["ops"]),
                "kernel_invocation_count": invocation_count,
            }
        )
        if context.options.level == "instruction":
            self._build_instruction_artifacts(context, prepared)
        metrics_path = context.artifact_dir / "hygon-metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        context.add_artifact(
            "normalized_metrics", "json", metrics_path, "application/json"
        )
        context.warnings.append(
            f"{prepared['tool']} timing is diagnostic and is not evaluator latency"
        )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError(
                "Hygon profile contains no device kernel invocation"
            )
        if (
            context.options.level == "instruction"
            and context.summary.get("instruction_count", 0) < 1
        ):
            raise ProfilerExecutionError(
                "Hygon code objects yielded no instruction listing"
            )


__all__ = ["HipprofProfiler"]
