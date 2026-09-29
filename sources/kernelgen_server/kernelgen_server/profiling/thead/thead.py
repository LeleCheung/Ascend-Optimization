"""T-Head ACU profiling for evaluator-owned commands."""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

from ..base import (
    ManagedProfiler,
    ProfileExecutionContext,
    ProfilerExecutionError,
    ProfilerUnsupportedError,
)
from ..models import ProfileOptions
from ..resource import host_global_lock
from ..source_view import report_source_target
from .thead_report import THeadReportMixin


AMPERFD_PATH = "/run/amperf/bin/amperfd"
AMPERFD_MAX_PAUSE_SEC = 600
THEAD_MAX_TIMEOUT_SEC = 590
THEAD_PCM_LOCK_PATH = "/run/amperf/kernelgen_thead_pcm.lock"
AMPERFD_COOLDOWN_RE = re.compile(
    r"retry after about (\d+)\s*second", re.IGNORECASE
)
AMPERFD_COOLDOWN_MAX_RETRIES = 3
AMPERFD_COOLDOWN_BUFFER_SEC = 5
AMPERFD_COOLDOWN_DEFAULT_SEC = 60


_ALLOWED_OPTIONS = {
    "acu_path",
    "amperfd_path",
    "ppu_sdk_root",
    "hgobjdump_path",
    "set",
    "launch_count",
    "sections",
    "metrics",
    "kernel_name",
    "kernel_name_base",
}


class THeadACUProfiler(THeadReportMixin, ManagedProfiler):
    backend = "thead"
    name = "acu"

    @staticmethod
    def _get_acu_path() -> str:
        return "acu" if shutil.which("acu") else "/usr/local/PPU_SDK/asight/bin/acu"

    @staticmethod
    def _tool_available(path: str) -> bool:
        return shutil.which(path) is not None or (
            Path(path).is_file() and os.access(path, os.X_OK)
        )

    def available(self) -> bool:
        return self._tool_available(self._get_acu_path())

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("thead", {}).get(
            "acu_path", self._get_acu_path()
        )
        return self._tool_available(str(configured))

    def capabilities(self) -> List[str]:
        return [
            "performance_counters",
            "memory_analysis",
            "occupancy",
            "source_mapping",
            "instruction_listing",
        ]

    def levels(self) -> List[str]:
        return ["metrics", "instruction"]

    def levels_for(self, options: ProfileOptions) -> List[str]:
        configured = options.backend_options.get("thead", {})
        if not self.available_for(options):
            return []
        levels = ["metrics"]
        ppu_sdk = Path(
            str(configured.get("ppu_sdk_root", "/usr/local/PPU_SDK"))
        ).resolve()
        hgobjdump = str(
            configured.get("hgobjdump_path", ppu_sdk / "bin" / "hgobjdump")
        )
        if self._tool_available(hgobjdump):
            levels.append("instruction")
        return levels

    def describe(self) -> Dict[str, Any]:
        result = super().describe()
        result["max_timeout_sec"] = THEAD_MAX_TIMEOUT_SEC
        return result

    def normalize_options(self, options):
        if options.timeout_sec <= THEAD_MAX_TIMEOUT_SEC:
            return options, []
        normalized = options.model_copy(
            update={"timeout_sec": THEAD_MAX_TIMEOUT_SEC}
        )
        return normalized, [
            f"thead timeout_sec clamped from {options.timeout_sec} to "
            f"{THEAD_MAX_TIMEOUT_SEC}; amperfd grants one 600s PCM pause window"
        ]

    @staticmethod
    def _wrap_with_envsetup(command: List[str], envsetup: Path) -> List[str]:
        quoted = " ".join(shlex.quote(str(item)) for item in command)
        return ["bash", "-c", f"source {shlex.quote(str(envsetup))} && exec {quoted}"]

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("thead", {})
        unknown = sorted(set(options) - _ALLOWED_OPTIONS)
        if unknown:
            raise ProfilerExecutionError(
                "unsupported T-Head profiler options: " + ", ".join(unknown)
            )
        executable = str(options.get("acu_path", self._get_acu_path()))
        if not self._tool_available(executable):
            raise ProfilerUnsupportedError(
                f"ACU executable was not found at {executable!r}"
            )
        ppu_sdk = Path(
            str(options.get("ppu_sdk_root", "/usr/local/PPU_SDK"))
        ).resolve()
        hgobjdump = str(
            options.get("hgobjdump_path", ppu_sdk / "bin" / "hgobjdump")
        )
        if context.options.level == "instruction" and not self._tool_available(
            hgobjdump
        ):
            raise ProfilerUnsupportedError(
                f"hgobjdump was not found at {hgobjdump!r}"
            )
        try:
            launch_count = max(1, int(options.get("launch_count", 1)))
        except (TypeError, ValueError) as exc:
            raise ProfilerExecutionError(
                "thead.launch_count must be an integer"
            ) from exc
        cache = context.artifact_dir / "triton-cache"
        if context.options.level == "instruction":
            cache.mkdir(exist_ok=False)
        return {
            "options": options,
            "executable": executable,
            "ppu_sdk": ppu_sdk,
            "envsetup": ppu_sdk / "envsetup.sh",
            "hgobjdump": hgobjdump,
            "amperfd": str(options.get("amperfd_path", AMPERFD_PATH)),
            "report_base": context.artifact_dir / "report",
            "report": context.artifact_dir / "report.acurep",
            "csv": context.artifact_dir / "report.csv",
            "cache": cache,
            "launch_count": launch_count,
        }

    def _amperfd(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        *args: str,
        label: str,
    ):
        return context.run_tool(
            [prepared["amperfd"], "profmetric", *args],
            label=label,
        )

    def _pause_collector(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> None:
        amperfd = prepared["amperfd"]
        if not self._tool_available(amperfd):
            raise ProfilerUnsupportedError(
                f"amperfd is required but not executable at {amperfd!r}"
            )
        last = ""
        for attempt in range(AMPERFD_COOLDOWN_MAX_RETRIES + 1):
            result = self._amperfd(
                context,
                prepared,
                "--pause",
                "-t",
                str(AMPERFD_MAX_PAUSE_SEC),
                label="T-Head pause amperfd",
            )
            output = result.output.strip()
            if result.returncode == 0 and "pause" in output.casefold():
                context.warnings.append(
                    "paused amperf collector for the ACU PCM window"
                )
                return
            last = output or f"exit status {result.returncode}"
            cooldown = AMPERFD_COOLDOWN_RE.search(last)
            if cooldown is None or attempt == AMPERFD_COOLDOWN_MAX_RETRIES:
                break
            wait = min(
                int(cooldown.group(1)) + AMPERFD_COOLDOWN_BUFFER_SEC,
                AMPERFD_COOLDOWN_DEFAULT_SEC + AMPERFD_COOLDOWN_BUFFER_SEC,
            )
            remaining = context.deadline.remaining("T-Head amperfd cooldown")
            time.sleep(min(float(wait), remaining))
        raise ProfilerExecutionError(f"could not pause amperfd: {last}")

    def _collector_is_running(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> tuple[bool, str]:
        result = self._amperfd(
            context,
            prepared,
            "--status",
            label="T-Head amperfd status",
        )
        return (
            result.returncode == 0 and "collecting" in result.output.casefold(),
            result.output.strip(),
        )

    def _resume_collector(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> None:
        result = self._amperfd(
            context,
            prepared,
            "--resume",
            label="T-Head resume amperfd",
        )
        last = result.output.strip()
        for _ in range(5):
            collecting, status = self._collector_is_running(context, prepared)
            last = status or last
            if collecting:
                return
            remaining = context.deadline.remaining("T-Head amperfd recovery")
            time.sleep(min(0.2, remaining))
        raise ProfilerExecutionError(
            "could not confirm amperfd collector recovery: " + last
        )

    @contextmanager
    def resource_scope(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> Iterator[None]:
        with host_global_lock(Path(THEAD_PCM_LOCK_PATH), context.deadline) as waited:
            context.summary["pcm_lock_wait_seconds"] = round(waited, 6)
            self._pause_collector(context, prepared)
            try:
                yield
            finally:
                self._resume_collector(context, prepared)

    @staticmethod
    def _option_values(value: Any) -> list[Any]:
        if value in (None, ""):
            return []
        return value if isinstance(value, list) else [value]

    def _acu_command(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> list[str]:
        options = prepared["options"]
        physical = context.command.env.get("CUDA_VISIBLE_DEVICES", "0")
        command = [
            prepared["executable"],
            "-o",
            str(prepared["report_base"]),
            "-f",
            "--devices",
            physical,
            "--profile-from-start",
            "no",
            "--launch-count",
            str(prepared["launch_count"]),
            "--kill",
            "no",
            "--check-exit-code",
            "yes",
        ]
        sections = self._option_values(options.get("sections"))
        if sections:
            for section in sections:
                command.extend(["--section", str(section)])
        elif options.get("metrics"):
            command.extend(["--metrics", str(options["metrics"])])
        else:
            command.extend(
                [
                    "--set",
                    str(
                        options.get(
                            "set",
                            "detailed"
                            if context.options.level == "metrics"
                            else "full",
                        )
                    ),
                ]
            )
        if context.options.level == "instruction":
            command.extend(["--import-source", "yes"])
        for key, flag in (
            ("kernel_name", "--kernel-name"),
            ("kernel_name_base", "--kernel-name-base"),
        ):
            if options.get(key) not in (None, ""):
                command.extend([flag, str(options[key])])
        command.extend(context.command.argv)
        if prepared["envsetup"].is_file():
            return self._wrap_with_envsetup(command, prepared["envsetup"])
        return command

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> Path:
        overrides = {"PPU_SDK": str(prepared["ppu_sdk"])}
        if context.options.level == "instruction":
            overrides.update(
                {
                    "TRITON_CACHE_DIR": str(prepared["cache"]),
                    "TRITON_DISABLE_LINE_INFO": "false",
                    "TRITON_ALWAYS_COMPILE": "1",
                }
            )
        execution = context.run_profile_stage(
            self._acu_command(context, prepared),
            label="T-Head ACU collection",
            env_overrides=overrides,
        )
        log = context.artifact_dir / "acu.log"
        log.write_text(execution.output, encoding="utf-8")
        context.add_artifact("profile_log", "text", log, "text/plain")
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"ACU exited with status {execution.returncode}"
            )
        if "Device is not ready for profiling" in execution.output:
            raise ProfilerExecutionError(
                "ACU device is not ready for profiling; PCM is unavailable"
            )
        if not prepared["report"].is_file() or prepared["report"].stat().st_size == 0:
            raise ProfilerExecutionError(
                "ACU completed without a non-empty report.acurep"
            )
        context.add_artifact("vendor_report", "acurep", prepared["report"])
        return prepared["report"]

    def _export_report(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> None:
        command = [
            prepared["executable"],
            "-i",
            str(prepared["report"]),
            "--csv",
            "--page",
            "details",
        ]
        if prepared["envsetup"].is_file():
            command = self._wrap_with_envsetup(command, prepared["envsetup"])
        execution = context.run_tool(
            command,
            label="T-Head ACU CSV export",
            env_overrides={"PPU_SDK": str(prepared["ppu_sdk"])},
        )
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"ACU CSV export exited with status {execution.returncode}"
            )
        lines = execution.output.splitlines(keepends=True)
        start = next(
            (
                index
                for index, line in enumerate(lines)
                if line.lstrip().startswith('"ID"')
            ),
            None,
        )
        if start is None:
            raise ProfilerExecutionError(
                "ACU CSV export did not contain the expected ID header"
            )
        prepared["csv"].write_text("".join(lines[start:]), encoding="utf-8")
        context.add_artifact("metrics_table", "csv", prepared["csv"], "text/csv")

    def _instruction_artifacts(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        kernel_names: set[str],
    ) -> None:
        binaries = list(self._iter_binaries(prepared["cache"]))
        matched = [
            binary
            for binary in binaries
            if not kernel_names
            or any(
                binary.stem == name
                or binary.stem in name
                or name in binary.stem
                for name in kernel_names
            )
        ]
        target = report_source_target(context.source_roots)
        source_rows: list[dict[str, Any]] = []
        instructions: list[dict[str, Any]] = []
        raw_chunks: list[str] = []
        disassembled = 0
        for index, binary in enumerate(matched):
            context.add_artifact("device_binary", binary.suffix.lstrip("."), binary)
            result = context.run_tool(
                [
                    prepared["hgobjdump"],
                    "--dump-isa",
                    "--line-numbers",
                    str(binary),
                ],
                label=f"T-Head hgobjdump {index}",
                env_overrides={"PPU_SDK": str(prepared["ppu_sdk"])},
            )
            if result.returncode != 0:
                context.warnings.append(f"hgobjdump failed for {binary.name}")
                continue
            disassembled += 1
            raw_chunks.extend(
                [f"===== {binary.name} =====", result.output.rstrip(), ""]
            )
            sources, rows = self._parse_objdump(result.output, binary)
            source_rows.extend(sources)
            instructions.extend(rows)
        source_rows, instructions = self._normalize_source_evidence(
            source_rows, instructions, target
        )
        raw = context.artifact_dir / "isa-dump.txt"
        raw.write_text("\n".join(raw_chunks).rstrip() + "\n", encoding="utf-8")
        context.add_artifact("isa_dump", "text", raw, "text/plain")
        source_payload = self._source_mapping_payload(
            source_rows, instructions, target
        )
        mapped = sum(
            len(item["instructions"])
            for item in source_payload["source_locations"]
        )
        listing = context.artifact_dir / "instructions.json"
        listing.write_text(
            json.dumps(
                {
                    "evidence_type": "static_instruction_listing",
                    "dynamic_timeline": False,
                    "instruction_count": len(instructions),
                    "mapped_instruction_count": mapped,
                    "instructions": instructions,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        context.add_artifact(
            "instruction_listing", "json", listing, "application/json"
        )
        summary = context.artifact_dir / "instruction-summary.txt"
        summary.write_text(
            self._render_instruction_summary(
                instructions,
                binary_count=len(matched),
                disassembled_binary_count=disassembled,
            ),
            encoding="utf-8",
        )
        context.add_artifact("instruction_summary", "text", summary, "text/plain")
        if mapped:
            source_map = context.artifact_dir / "source-map.json"
            source_map.write_text(json.dumps(source_payload, indent=2), encoding="utf-8")
            context.add_artifact(
                "source_mapping", "json", source_map, "application/json"
            )
            context.add_capability("source_mapping")
        elif instructions:
            context.warnings.append(
                "T-Head instruction listing contains no source mapping"
            )
        context.summary.update(
            {
                "binary_count": len(matched),
                "disassembled_binary_count": disassembled,
                "instruction_count": len(instructions),
                "mapped_instruction_count": mapped,
            }
        )
        if instructions:
            context.add_capability("instruction_listing")

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: Path,
    ) -> None:
        del collected
        self._export_report(context, prepared)
        try:
            metrics = self._parse_report_csv(prepared["csv"])
        except (OSError, csv.Error, ValueError) as exc:
            raise ProfilerExecutionError(f"could not parse ACU CSV: {exc}") from exc
        if not metrics.get("kernels"):
            raise ProfilerExecutionError("ACU CSV contained no kernel metrics")
        kernel_names = {
            str(item.get("name", ""))
            for item in metrics["kernels"]
            if str(item.get("name", ""))
        }
        metrics.update(
            {
                "kernel_invocation_count": len(metrics["kernels"]),
                "operation_count": len(kernel_names),
                "measured_iterations": context.options.iterations,
            }
        )
        context.metrics = metrics
        context.summary.update(
            {
                "kernel_count": len(metrics["kernels"]),
                "kernel_invocation_count": metrics["kernel_invocation_count"],
                "operation_count": metrics["operation_count"],
                "profiled_device": context.command.env.get(
                    "CUDA_VISIBLE_DEVICES", "0"
                ),
            }
        )
        context.add_capability(
            "performance_counters", "memory_analysis", "occupancy"
        )
        metrics_path = context.artifact_dir / "acu-metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        context.add_artifact(
            "normalized_metrics", "json", metrics_path, "application/json"
        )
        details = context.artifact_dir / "profile-details.txt"
        details.write_text(
            self._render_profile_details(
                metrics,
                context.summary,
                device=context.device,
                warmup=context.options.warmup,
                iterations=context.options.iterations,
            ),
            encoding="utf-8",
        )
        context.add_artifact("profile_details", "text", details, "text/plain")
        if context.options.level == "instruction":
            self._instruction_artifacts(context, prepared, kernel_names)
        context.warnings.append(
            "ACU instrumented timing is diagnostic and is not evaluator latency"
        )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError("ACU metrics contain no kernel records")
        if (
            context.options.level == "instruction"
            and context.summary.get("instruction_count", 0) < 1
        ):
            raise ProfilerExecutionError(
                "PPU binaries yielded no instruction listing"
            )


__all__ = ["THeadACUProfiler"]
