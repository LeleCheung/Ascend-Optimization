"""Enflame topsprof and source-mapped GCU disassembly."""

from __future__ import annotations

import csv
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
from ..source_view import report_source_target
from .enflame_report import (
    _BUNDLE_ARCH_RE,
    _render_instruction_report,
    _render_instruction_summary,
    _render_source_report,
    _source_mapping_payload,
    parse_instruction_dump,
    parse_topsprof_summary,
    parse_topsprof_trace,
)


def _tool_available(executable: str) -> bool:
    path = Path(executable)
    return path.is_file() if path.is_absolute() else shutil.which(executable) is not None


class TopsProfProfiler(ManagedProfiler):
    backend = "enflame"
    name = "topsprof+llvm-objdump"

    def available(self) -> bool:
        return _tool_available("topsprof") or _tool_available("/opt/tops/bin/topsprof")

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("enflame", {}).get(
            "topsprof_path", "/opt/tops/bin/topsprof"
        )
        return _tool_available(str(configured))

    def levels(self) -> List[str]:
        levels = ["metrics"]
        if all(
            _tool_available(path)
            for path in (
                "/opt/tops/bin/clang-offload-bundler",
                "/opt/tops/bin/llvm-objdump",
            )
        ):
            levels.append("instruction")
        return levels

    def levels_for(self, options: ProfileOptions) -> List[str]:
        configured = options.backend_options.get("enflame", {})
        if not self.available_for(options):
            return []
        levels = ["metrics"]
        bundler = str(
            configured.get("bundler_path", "/opt/tops/bin/clang-offload-bundler")
        )
        objdump = str(configured.get("objdump_path", "/opt/tops/bin/llvm-objdump"))
        if _tool_available(bundler) and _tool_available(objdump):
            levels.append("instruction")
        return levels

    def capabilities(self) -> List[str]:
        return [
            "kernel_profile", "execution_timeline", "device_binary",
            "device_object", "compiler_ir", "source_mapping", "instruction_listing",
        ]

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("enflame", {})
        allowed = {"topsprof_path", "activities", "bundler_path", "objdump_path"}
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ProfilerExecutionError(
                f"unsupported Enflame profiler options: {', '.join(unknown)}"
            )
        executable = str(options.get("topsprof_path", "/opt/tops/bin/topsprof"))
        if not _tool_available(executable):
            raise ProfilerUnsupportedError(
                f"topsprof executable was not found at {executable!r}"
            )
        bundler = str(
            options.get("bundler_path", "/opt/tops/bin/clang-offload-bundler")
        )
        objdump = str(options.get("objdump_path", "/opt/tops/bin/llvm-objdump"))
        if context.options.level == "instruction":
            missing = [tool for tool in (bundler, objdump) if not _tool_available(tool)]
            if missing:
                raise ProfilerUnsupportedError(
                    "Enflame instruction tools were not found: " + ", ".join(missing)
                )
        cache = context.artifact_dir / "triton-cache"
        cache.mkdir(exist_ok=False)
        return {
            "options": options,
            "executable": executable,
            "bundler": bundler,
            "objdump": objdump,
            "cache": cache,
            "vpd": context.artifact_dir / "topsprof.vpd",
            "csv": context.artifact_dir / "topsprof-summary.csv",
        }

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        app_log = context.artifact_dir / "topsprof-app.log"
        command = [
            prepared["executable"], "--enable-activities",
            str(prepared["options"].get("activities", "operator,memcpy")),
            "--export", str(prepared["vpd"]), "--export-csv", str(prepared["csv"]),
            "--print-gcu-summary", "--timeunit", "usec", "--force-overwrite",
            "--log-file", str(app_log), "--profile-from-start", "off",
            "--capture-range", "topsProfilerApi", "--kill", "none",
            *context.command.argv,
        ]
        overrides = {
            "TRITON_CACHE_DIR": str(prepared["cache"]),
            "KGS_CACHE_PATH": str(context.artifact_dir / "module-cache"),
            "TRITON_DISABLE_LINE_INFO": (
                "0" if context.options.level == "instruction" else "1"
            ),
            "KGS_ENFLAME_LINE_INFO": (
                "1" if context.options.level == "instruction" else "0"
            ),
        }
        if context.options.level == "instruction":
            overrides["LLVM_EXTRACT_DI_LOCAL_VARIABLES"] = "1"
        execution = context.run_profile_stage(
            command, label="Enflame topsprof collection", env_overrides=overrides
        )
        raw_log = context.artifact_dir / "topsprof.log"
        raw_log.write_text(execution.output, encoding="utf-8")
        context.add_artifact("profile_log", "text", raw_log, "text/plain")
        if app_log.is_file():
            context.add_artifact("profile_log", "text", app_log, "text/plain")
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"topsprof exited with status {execution.returncode}"
            )
        if not prepared["vpd"].is_file() or prepared["vpd"].stat().st_size == 0:
            raise ProfilerExecutionError("topsprof did not produce a non-empty VPD")
        if not prepared["csv"].is_file() or prepared["csv"].stat().st_size == 0:
            raise ProfilerExecutionError("topsprof did not produce a non-empty summary CSV")
        context.add_artifact("vendor_report", "vpd", prepared["vpd"])
        context.add_artifact("metrics_table", "csv", prepared["csv"], "text/csv")
        return {}

    def _instruction_artifacts(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        target = report_source_target(context.source_roots)
        binaries = sorted(prepared["cache"].rglob("*.fatbin"))
        object_dir = context.artifact_dir / "device-objects"
        object_dir.mkdir(exist_ok=False)
        instructions: list[dict[str, Any]] = []
        raw_chunks: list[str] = []
        disassembled = 0
        seen_ir: set[Path] = set()
        for index, fatbin in enumerate(binaries):
            context.add_artifact("device_binary", "gcu-fatbin", fatbin)
            listed = context.run_tool(
                [prepared["bundler"], "--list", "--type=o", f"--input={fatbin}"],
                label=f"Enflame list fatbin {index}",
            )
            targets = [
                line.strip()
                for line in listed.output.splitlines()
                if line.strip().startswith("tops-") and "gcu" in line
            ]
            if listed.returncode != 0 or not targets:
                context.warnings.append(f"no GCU offload target found in {fatbin.name}")
                continue
            bundle_target = targets[0]
            arch_match = _BUNDLE_ARCH_RE.search(bundle_target)
            arch = arch_match.group(1) if arch_match else "gcu300"
            obj = object_dir / f"{index:03d}-{fatbin.stem}.gcu.o"
            unpacked = context.run_tool(
                [prepared["bundler"], "--unbundle", "--type=o",
                 f"--targets={bundle_target}", f"--input={fatbin}", f"--output={obj}"],
                label=f"Enflame unbundle {index}",
            )
            if unpacked.returncode != 0 or not obj.is_file():
                context.warnings.append(f"could not unpack {fatbin.name}")
                continue
            dump = context.run_tool(
                [prepared["objdump"], "-S", "-l", "-C", "-d", f"--mcpu={arch}", str(obj)],
                label=f"Enflame disassemble {index}",
            )
            if dump.returncode != 0:
                context.warnings.append(f"llvm-objdump failed for {fatbin.name}")
                continue
            disassembled += 1
            context.add_artifact("device_object", "elf64-dtu", obj)
            raw_chunks.extend(
                [f"===== {fatbin.name} ({bundle_target}) =====", dump.output.rstrip(), ""]
            )
            instructions.extend(
                parse_instruction_dump(
                    dump.output, binary_name=fatbin.name, target=target
                )
            )
            for ir in sorted(fatbin.parent.iterdir()):
                if ir in seen_ir or ir.suffix not in {
                    ".ttir", ".ttgir", ".gcuir", ".llir", ".source"
                }:
                    continue
                seen_ir.add(ir)
                context.add_artifact("compiler_ir", ir.suffix[1:], ir, "text/plain")
        raw = context.artifact_dir / "isa-dump.txt"
        raw.write_text("\n".join(raw_chunks).rstrip() + "\n", encoding="utf-8")
        context.add_artifact("isa_dump", "text", raw, "text/plain")
        source_mapping = _source_mapping_payload(instructions, target)
        payload = {
            "evidence_type": "static_instruction_listing",
            "instruction_count": len(instructions),
            "mapped_instruction_count": source_mapping["mapped_instruction_count"],
            "instructions": instructions,
        }
        listing = context.artifact_dir / "instructions.json"
        listing.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mapping = context.artifact_dir / "source-map.json"
        mapping.write_text(json.dumps(source_mapping, indent=2), encoding="utf-8")
        summary_text = _render_instruction_summary(
            instructions, source_mapping, binary_count=disassembled
        )
        summary = context.artifact_dir / "instruction-summary.txt"
        summary.write_text(summary_text, encoding="utf-8")
        report = context.artifact_dir / "instruction-report.txt"
        report.write_text(_render_instruction_report(instructions), encoding="utf-8")
        source_report = context.artifact_dir / "source-report.txt"
        source_report.write_text(_render_source_report(source_mapping), encoding="utf-8")
        for kind, format_name, path, media in (
            ("instruction_listing", "json", listing, "application/json"),
            ("source_mapping", "json", mapping, "application/json"),
            ("instruction_summary", "text", summary, "text/plain"),
            ("instruction_report", "text", report, "text/plain"),
            ("source_report", "text", source_report, "text/plain"),
        ):
            context.add_artifact(kind, format_name, path, media)
        return {
            "binary_count": len(binaries),
            "disassembled_binary_count": disassembled,
            "instruction_count": len(instructions),
            "mapped_instruction_count": source_mapping["mapped_instruction_count"],
            "source_location_count": source_mapping["source_location_count"],
        }

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: dict[str, Any],
    ) -> None:
        del collected
        try:
            metrics = parse_topsprof_summary(prepared["csv"])
        except (OSError, csv.Error, ValueError) as exc:
            raise ProfilerExecutionError(f"could not parse topsprof summary: {exc}") from exc
        metrics["measured_iterations"] = context.options.iterations
        trace = context.run_tool(
            [prepared["executable"], "--import", str(prepared["vpd"]),
             "--print-gcu-trace", "--timeunit", "usec"],
            label="topsprof trace export",
        )
        if trace.returncode == 0:
            timeline = parse_topsprof_trace(trace.output)
            if timeline:
                metrics["timeline"] = timeline
                timeline_path = context.artifact_dir / "topsprof-trace.txt"
                timeline_path.write_text(trace.output, encoding="utf-8")
                context.add_artifact(
                    "execution_timeline", "text", timeline_path, "text/plain"
                )
                context.add_capability("execution_timeline")
        else:
            context.warnings.append(
                f"topsprof trace export exited with status {trace.returncode}"
            )
        context.metrics = metrics
        context.summary.update(
            {
                "kernel_count": metrics["kernel_count"],
                "kernel_invocation_count": metrics["kernel_invocation_count"],
                "total_kernel_time_us": metrics["total_kernel_time_us"],
            }
        )
        context.add_capability("kernel_profile")
        if context.options.level == "instruction":
            context.summary.update(self._instruction_artifacts(context, prepared))
            context.add_capability("instruction_listing", "source_mapping")
            for artifact_kind in ("device_binary", "device_object", "compiler_ir"):
                if context.has_artifact(artifact_kind):
                    context.add_capability(artifact_kind)
        metrics_path = context.artifact_dir / "topsprof-metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        context.add_artifact(
            "normalized_metrics", "json", metrics_path, "application/json"
        )
        context.warnings.extend(
            [
                "topsprof timing is diagnostic and is not eval latency",
                "topsprof 1.9 does not expose hardware performance counters",
            ]
        )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError(
                "topsprof summary contains no GCU activity"
            )
        if context.options.level == "instruction":
            if context.summary.get("instruction_count", 0) < 1:
                raise ProfilerExecutionError(
                    "GCU binaries did not yield an instruction listing"
                )
            if context.summary.get("mapped_instruction_count", 0) < 1:
                raise ProfilerExecutionError(
                    "GCU instruction listing has no mapping to submitted source"
                )


__all__ = ["TopsProfProfiler"]
