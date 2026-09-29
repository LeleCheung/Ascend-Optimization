# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Huawei Ascend msprof profiling for evaluator-owned commands."""

from __future__ import annotations

import csv
import json
import math
import platform
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..base import (
    ManagedProfiler,
    ProfileExecutionContext,
    ProfilerExecutionError,
    ProfilerUnsupportedError,
)
from ..models import ProfileOptions
from .ascend_instruction import build_instruction_report, render_instruction_details


_IGNORED_OP_PATTERN = re.compile(r"aclnnIsClose|aclnnAll_ReduceAll")
_SKIPPED_KERNEL_PATTERN = re.compile(
    r"Kernel (.+?) skipped: not selected via --kernel-name\."
)
_OP_METRIC_COLUMNS = {
    "aicore_time(us)": "aicore_time_us",
    "aic_total_cycles": "aic_total_cycles",
    "aic_mac_time(us)": "aic_mac_time_us",
    "aic_mac_ratio": "aic_mac_ratio",
    "aic_scalar_time(us)": "aic_scalar_time_us",
    "aic_scalar_ratio": "aic_scalar_ratio",
    "aic_mte1_time(us)": "aic_mte1_time_us",
    "aic_mte1_ratio": "aic_mte1_ratio",
    "aic_mte2_time(us)": "aic_mte2_time_us",
    "aic_mte2_ratio": "aic_mte2_ratio",
    "aic_fixpipe_time(us)": "aic_fixpipe_time_us",
    "aic_fixpipe_ratio": "aic_fixpipe_ratio",
    "aic_icache_miss_rate": "aic_icache_miss_rate",
    "aiv_time(us)": "aiv_time_us",
    "aiv_total_cycles": "aiv_total_cycles",
    "aiv_vec_time(us)": "aiv_vec_time_us",
    "aiv_vec_ratio": "aiv_vec_ratio",
    "aiv_scalar_time(us)": "aiv_scalar_time_us",
    "aiv_scalar_ratio": "aiv_scalar_ratio",
    "aiv_mte2_time(us)": "aiv_mte2_time_us",
    "aiv_mte2_ratio": "aiv_mte2_ratio",
    "aiv_mte3_time(us)": "aiv_mte3_time_us",
    "aiv_mte3_ratio": "aiv_mte3_ratio",
    "aiv_icache_miss_rate": "aiv_icache_miss_rate",
    "cube_utilization(%)": "cube_utilization_pct",
}


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_op_summary(path: Path, warmup: int, iterations: int) -> Dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("op_summary CSV is empty")
    name_col = "Op Name" if "Op Name" in rows[0] else next(iter(rows[0]))
    rows = [row for row in rows if not _IGNORED_OP_PATTERN.search(row.get(name_col, ""))]
    counts = Counter(row.get(name_col, "") for row in rows)
    expected = warmup + iterations
    valid_names = {name for name, count in counts.items() if count == expected}
    if not valid_names:
        valid_names = {name for name, count in counts.items() if count >= iterations}
    if not valid_names:
        raise ValueError("no profiled ops have the expected invocation count")
    time_col = next(
        (name for name in rows[0] if "Duration" in name), "Task Duration(us)"
    )
    by_name: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get(name_col, "") in valid_names:
            by_name[row[name_col]].append(row)
    operations: list[dict[str, Any]] = []
    for name, op_rows in by_name.items():
        measured = op_rows[warmup:] if len(op_rows) > warmup else op_rows[-iterations:]
        durations = [
            value
            for row in measured
            if (value := _as_float(row.get(time_col))) is not None
        ]
        if not durations:
            continue
        sample = measured[0]
        detail: dict[str, Any] = {
            "op_name": name,
            "op_type": sample.get("OP Type", ""),
            "task_type": sample.get("Task Type", ""),
            "measured_invocations": len(durations),
            "total_duration_us": round(sum(durations), 3),
            "avg_duration_us": round(sum(durations) / len(durations), 3),
            "min_duration_us": round(min(durations), 3),
            "max_duration_us": round(max(durations), 3),
        }
        for column, key in _OP_METRIC_COLUMNS.items():
            values = [
                value
                for row in measured
                if (value := _as_float(row.get(column))) is not None
            ]
            if values:
                detail[key] = round(sum(values) / len(values), 4)
        operations.append(detail)
    if not operations:
        raise ValueError("op_summary CSV contained no measurable operations")
    return {
        "avg_time_us": round(sum(op["avg_duration_us"] for op in operations), 3),
        "ops": operations,
    }


def _first_matching(root: Path, pattern: str) -> Optional[Path]:
    return next(iter(root.rglob(pattern)), None)


def _discovered_kernel_names(log_text: str, expected: int, minimum: int) -> list[str]:
    counts = Counter(_SKIPPED_KERNEL_PATTERN.findall(log_text))
    exact = sorted(name for name, count in counts.items() if count == expected)
    return exact or sorted(name for name, count in counts.items() if count >= minimum)


def _configured_kernel_names(options: Mapping[str, Any]) -> list[str]:
    value = options.get("op_kernel_names", options.get("op_kernel_name"))
    if isinstance(value, str):
        names = [value] if value else []
    elif isinstance(value, list):
        names = [str(name) for name in value if name]
    else:
        names = []
    return list(dict.fromkeys(names))


def _find_simulator_lib(
    soc_version: str, options: Mapping[str, Any], env: Mapping[str, str]
) -> Optional[Path]:
    configured = options.get("simulator_lib_path")
    if configured:
        path = Path(str(configured))
        return path if path.is_dir() else None
    architecture = platform.machine()
    roots = [
        env.get("ASCEND_HOME_PATH"),
        env.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/cann",
        "/usr/local/Ascend/ascend-toolkit/latest",
    ]
    for root_value in roots:
        if not root_value:
            continue
        root = Path(root_value)
        for candidate in (
            root / f"{architecture}-linux" / "simulator" / soc_version / "lib",
            root / "tools" / "simulator" / soc_version / "lib",
        ):
            if candidate.is_dir():
                return candidate
    return None


def _source_files(context: ProfileExecutionContext) -> dict[str, str]:
    result: dict[str, str] = {}
    for root in context.source_roots:
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                result[path.relative_to(root).as_posix()] = content
    return result


class MsprofProfiler(ManagedProfiler):
    backend = "npu"
    name = "msprof"

    def available(self) -> bool:
        return shutil.which("msprof") is not None

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("npu", {}).get(
            "msprof_path", "msprof"
        )
        return shutil.which(str(configured)) is not None

    def capabilities(self) -> List[str]:
        return [
            "performance_counters",
            "pipeline_utilization",
            "kernel_profile",
            "instruction_listing",
        ]

    def levels(self) -> List[str]:
        return ["metrics", "instruction"]

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("npu", {})
        allowed = {
            "msprof_path", "ai_core", "aic_metrics", "op_detail", "op_metrics",
            "op_launch_count", "op_kernel_name", "op_kernel_names", "simulator",
            "simulator_soc_version", "npu_smi_path", "simulator_lib_path",
            "simulator_core_ids", "simulator_timeout_min", "simulator_dump",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ProfilerExecutionError(
                f"unsupported npu profiler options: {', '.join(unknown)}"
            )
        executable = str(options.get("msprof_path", "msprof"))
        if shutil.which(executable) is None:
            raise ProfilerUnsupportedError(
                f"msprof executable was not found at {executable!r}"
            )
        prepared: dict[str, Any] = {
            "options": options,
            "executable": executable,
            "output_dir": context.artifact_dir / "msprof-output",
        }
        if context.options.level == "instruction":
            soc = str(options.get("simulator_soc_version", "")).strip()
            if not soc:
                smi = str(options.get("npu_smi_path", "npu-smi"))
                if shutil.which(smi) is not None:
                    physical = context.command.env.get("ASCEND_RT_VISIBLE_DEVICES", "0")
                    probe = context.run_tool(
                        [smi, "info", "-t", "board", "-i", physical, "-c", "0"],
                        label="Ascend SoC detection",
                    )
                    match = re.search(r"Chip Name\s*:\s*([^\s]+)", probe.output)
                    if probe.returncode == 0 and match:
                        raw = match.group(1)
                        soc = raw if raw.startswith("Ascend") else f"Ascend{raw}"
            if not soc:
                raise ProfilerUnsupportedError(
                    "Ascend simulator SoC could not be determined"
                )
            candidates = [soc]
            family, separator, revision = soc.rpartition("-")
            if separator and family and revision.isdigit():
                candidates.append(family)
            selected_soc = soc
            simulator_lib = None
            for candidate in dict.fromkeys(candidates):
                simulator_lib = _find_simulator_lib(
                    candidate, options, context.environment()
                )
                if simulator_lib is not None:
                    selected_soc = candidate
                    break
            if simulator_lib is None:
                raise ProfilerUnsupportedError(
                    f"Ascend simulator library was not found for {soc}"
                )
            prepared.update(
                {"soc_version": selected_soc, "simulator_lib": simulator_lib}
            )
        return prepared

    def _run_logged(
        self,
        context: ProfileExecutionContext,
        command: list[str],
        name: str,
        *,
        env_overrides: Mapping[str, str] | None = None,
        required: bool = True,
    ) -> Any:
        execution = context.run_profile_stage(
            command, label=name, env_overrides=env_overrides
        )
        log_path = context.artifact_dir / (
            re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") + ".log"
        )
        log_path.write_text(execution.output, encoding="utf-8")
        context.add_artifact("profile_log", "text", log_path, "text/plain")
        if execution.returncode != 0:
            message = f"{name} exited with status {execution.returncode}"
            if required:
                raise ProfilerExecutionError(message)
            context.warnings.append(message)
            return None
        return execution

    def _discover_kernels(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> list[str]:
        options = prepared["options"]
        configured = _configured_kernel_names(options)
        if configured:
            return configured
        output = context.artifact_dir / "msprof-op-discovery-output"
        command = [
            prepared["executable"], "op",
            "--kernel-name=__kgs_discovery_no_match__", "--launch-count=1",
            "--aic-metrics=BasicInfo", "--replay-mode=kernel",
            f"--output={output}", *context.command.argv,
        ]
        execution = self._run_logged(
            context, command, "msprof op kernel discovery", required=False
        )
        if execution is None:
            return []
        return _discovered_kernel_names(
            execution.output,
            context.options.warmup + context.options.iterations,
            context.options.iterations,
        )

    def _collect_op_detail(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        kernel_names: list[str],
    ) -> list[str]:
        options = prepared["options"]
        try:
            launch_count = int(options.get("op_launch_count", 1))
        except (TypeError, ValueError):
            launch_count = 1
            context.warnings.append("invalid npu.op_launch_count; using 1")
        launch_count = min(max(1, launch_count), context.options.iterations)
        metrics_value = options.get("op_metrics", "Default,BasicInfo")
        op_metrics = (
            ",".join(str(value) for value in metrics_value if value)
            if isinstance(metrics_value, list)
            else str(metrics_value or "Default,BasicInfo")
        )
        root = context.artifact_dir / "msprof-op-output"
        root.mkdir(exist_ok=False)
        collected: list[str] = []
        for index, kernel_name in enumerate(kernel_names):
            output = root / f"capture-{index:03d}"
            command = [
                prepared["executable"], "op", f"--kernel-name={kernel_name}",
                f"--warm-up={context.options.warmup}",
                f"--launch-count={launch_count}", f"--aic-metrics={op_metrics}",
                "--replay-mode=kernel", f"--output={output}",
                *context.command.argv,
            ]
            execution = self._run_logged(
                context, command, f"msprof op kernel {index}", required=False
            )
            if execution is not None and any(output.glob("OPPROF_*")):
                normalized = root / f"kernel-{len(collected):03d}"
                output.rename(normalized)
                collected.append(kernel_name)
            else:
                context.warnings.append(
                    f"msprof op produced no report for kernel {kernel_name!r}"
                )
        if collected:
            archive = Path(
                shutil.make_archive(
                    str(context.artifact_dir / "msprof-op-report"),
                    "gztar", root_dir=str(root),
                )
            )
            context.add_artifact(
                "vendor_kernel_report", "msprof-op-tar-gz", archive, "application/gzip"
            )
            context.summary["detailed_kernel_count"] = len(collected)
            context.metrics["detailed_kernel_names"] = collected
            context.add_capability(
                "kernel_profile", "arithmetic_utilization", "memory_counters",
                "cache_counters", "resource_conflicts",
            )
        return collected

    def _collect_simulator(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        kernel_names: list[str],
    ) -> dict[str, Any] | None:
        if not kernel_names:
            return None
        options = prepared["options"]
        core_ids_value = options.get("simulator_core_ids", "0")
        core_ids = (
            "|".join(str(value) for value in core_ids_value)
            if isinstance(core_ids_value, list)
            else str(core_ids_value or "0")
        )
        try:
            timeout_minutes = int(
                options.get(
                    "simulator_timeout_min",
                    max(1, math.ceil(context.deadline.remaining() / 60)),
                )
            )
        except (TypeError, ValueError):
            timeout_minutes = max(1, math.ceil(context.deadline.remaining() / 60))
        output_root = context.artifact_dir / "msprof-simulator-output"
        output_root.mkdir(exist_ok=False)
        env = context.environment()
        existing = env.get("LD_LIBRARY_PATH", "")
        library = str(prepared["simulator_lib"])
        env_overrides = {
            "LD_LIBRARY_PATH": f"{library}:{existing}" if existing else library,
            "TRITON_DISABLE_LINE_INFO": "false",
            "TRITON_CACHE_DIR": str(context.artifact_dir / "ascend-simulator-cache"),
        }
        collected: list[str] = []
        for index, kernel_name in enumerate(kernel_names):
            output = output_root / f"capture-{index:03d}"
            command = [
                prepared["executable"], "op", "simulator",
                f"--kernel-name={kernel_name}",
                f"--soc-version={prepared['soc_version']}", f"--core-id={core_ids}",
                f"--dump={'on' if options.get('simulator_dump', True) else 'off'}",
                f"--output={output}", f"--timeout={max(1, timeout_minutes)}",
                *context.command.argv,
            ]
            execution = self._run_logged(
                context, command, f"Ascend simulator kernel {index}",
                env_overrides=env_overrides, required=False,
            )
            reports = list(output.glob("OPPROF_*/simulator"))
            if execution is not None and reports:
                output.rename(output_root / f"kernel-{len(collected):03d}")
                collected.append(kernel_name)
            else:
                context.warnings.append(
                    f"Ascend simulator produced no report for kernel {kernel_name!r}"
                )
        if not collected:
            return None
        archive = Path(
            shutil.make_archive(
                str(context.artifact_dir / "msprof-simulator-report"),
                "gztar", root_dir=str(output_root),
            )
        )
        context.add_artifact(
            "vendor_instruction_report", "msprof-simulator-tar-gz", archive,
            "application/gzip",
        )
        report, warnings = build_instruction_report(
            output_root, collected, prepared["soc_version"], _source_files(context)
        )
        context.warnings.extend(warnings)
        listing = context.artifact_dir / "ascend-instructions.json"
        listing.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        context.add_artifact(
            "instruction_listing", "aicore-instruction-json", listing,
            "application/json",
        )
        return report

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        options = prepared["options"]
        command = [prepared["executable"], f"--output={prepared['output_dir']}"]
        if options.get("ai_core", True):
            command.append("--ai-core=on")
        if options.get("aic_metrics", "PipeUtilization"):
            command.append(f"--aic-metrics={options.get('aic_metrics', 'PipeUtilization')}")
        command.extend(context.command.argv)
        execution = self._run_logged(context, command, "msprof base collection")
        profile_root = prepared["output_dir"]
        for line in execution.output.splitlines():
            match = re.search(r"Data is saved in (.+?)$", line)
            if match and Path(match.group(1).strip()).is_dir():
                profile_root = Path(match.group(1).strip())
                break
        if not profile_root.is_dir():
            raise ProfilerExecutionError(
                "msprof completed but no profile output directory was found"
            )
        archive = Path(
            shutil.make_archive(
                str(context.artifact_dir / "msprof-report"),
                "gztar", root_dir=str(profile_root),
            )
        )
        context.add_artifact(
            "vendor_report", "msprof-tar-gz", archive, "application/gzip"
        )
        result: dict[str, Any] = {"profile_root": profile_root}
        if context.options.level == "instruction":
            kernels = self._discover_kernels(context, prepared)
            if options.get("op_detail", True):
                detailed = self._collect_op_detail(context, prepared, kernels)
                kernels = detailed or kernels
            report = self._collect_simulator(context, prepared, kernels)
            result["instruction_report"] = report
        return result

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: dict[str, Any],
    ) -> None:
        del prepared
        profile_root = collected["profile_root"]
        op_summary = _first_matching(profile_root, "op_summary_*.csv")
        if op_summary is None:
            raise ProfilerExecutionError(
                "msprof did not produce the required op_summary CSV"
            )
        copied = context.artifact_dir / "op_summary.csv"
        shutil.copy2(op_summary, copied)
        context.add_artifact("op_summary", "csv", copied, "text/csv")
        try:
            parsed = _parse_op_summary(
                op_summary, context.options.warmup, context.options.iterations
            )
        except Exception as exc:
            raise ProfilerExecutionError(
                f"could not parse required op_summary CSV: {exc}"
            ) from exc
        context.metrics.update(parsed)
        context.summary.update(
            {
                "avg_time_us": parsed["avg_time_us"],
                "operation_count": len(parsed["ops"]),
                "top_operations": sorted(
                    parsed["ops"],
                    key=lambda item: item["avg_duration_us"],
                    reverse=True,
                )[:10],
            }
        )
        context.add_capability("performance_counters", "kernel_profile")
        if any(
            any(key.endswith("_ratio") or key == "cube_utilization_pct" for key in op)
            for op in parsed["ops"]
        ):
            context.add_capability("pipeline_utilization")

        utilization = _first_matching(profile_root, "ai_core_utilization_*.csv")
        if utilization:
            copied_utilization = context.artifact_dir / "ai_core_utilization.csv"
            shutil.copy2(utilization, copied_utilization)
            context.add_artifact(
                "performance_counters", "csv", copied_utilization, "text/csv"
            )
        trace = _first_matching(profile_root, "msprof_*.json") or _first_matching(
            profile_root, "trace.json"
        )
        if trace:
            copied_trace = context.artifact_dir / "execution-trace.json"
            shutil.copy2(trace, copied_trace)
            context.add_artifact(
                "execution_timeline", "trace-json", copied_trace, "application/json"
            )
            context.add_capability("execution_timeline")

        report = collected.get("instruction_report")
        if report:
            cores = [
                core
                for kernel in report.get("kernels", [])
                for core in kernel.get("cores", [])
            ]
            instruction_count = sum(
                core.get("listed_instruction_count", 0) for core in cores
            )
            mapped_count = sum(core.get("mapped_instruction_count", 0) for core in cores)
            context.summary.update(
                {
                    "simulator_kernel_count": len(report.get("kernels", [])),
                    "simulator_core_count": len(cores),
                    "simulator_instruction_count": instruction_count,
                    "simulator_mapped_instruction_count": mapped_count,
                    "text": render_instruction_details(report)[:16_000],
                }
            )
            context.metrics["simulator"] = {
                "soc_version": report.get("soc_version", ""),
                "instruction_count": instruction_count,
                "mapped_instruction_count": mapped_count,
            }
            if instruction_count:
                context.add_capability("instruction_listing")
            if mapped_count:
                context.add_capability("source_hotspots")
        context.warnings.append(
            "msprof and simulator timing is diagnostic and is not eval latency"
        )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if not context.metrics.get("ops"):
            raise ProfilerExecutionError(
                "msprof op summary did not contain measurable operations"
            )
        if (
            context.options.level == "instruction"
            and context.summary.get("simulator_instruction_count", 0) < 1
        ):
            raise ProfilerExecutionError(
                "Ascend simulator did not produce an instruction listing"
            )


__all__ = ["MsprofProfiler", "_parse_op_summary"]
