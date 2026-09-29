"""MetaX mcTracer and MACA compiler evidence profiling."""

from __future__ import annotations

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
from .metax_report import (
    _hex_dump,
    _render_instruction_summary,
    _render_source_report,
    _source_mapping_payload,
    parse_mctracer_trace,
    parse_ttir_instructions,
)


_DEFAULT_MCTRACER = "/opt/maca/bin/mcTracer"
_DEFAULT_TOOLCHAIN = Path("/opt/maca/mxgpu_llvm/bin")


def _tool_available(executable: str) -> bool:
    path = Path(executable)
    return path.is_file() if path.is_absolute() else shutil.which(executable) is not None


class McTracerProfiler(ManagedProfiler):
    backend = "metax"
    name = "mcTracer+maca-llvm"

    def available(self) -> bool:
        return _tool_available(_DEFAULT_MCTRACER)

    def available_for(self, options: ProfileOptions) -> bool:
        configured = options.backend_options.get("metax", {}).get(
            "mctracer_path", _DEFAULT_MCTRACER
        )
        return _tool_available(str(configured))

    def levels(self) -> List[str]:
        levels = ["metrics"]
        if all(
            _tool_available(str(_DEFAULT_TOOLCHAIN / name))
            for name in ("clang-offload-bundler", "llvm-dis")
        ):
            levels.append("instruction")
        return levels

    def levels_for(self, options: ProfileOptions) -> List[str]:
        configured = options.backend_options.get("metax", {})
        if not self.available_for(options):
            return []
        levels = ["metrics"]
        toolchain = Path(str(configured.get("toolchain_path", _DEFAULT_TOOLCHAIN)))
        bundler = str(
            configured.get("bundler_path", toolchain / "clang-offload-bundler")
        )
        llvm_dis = str(configured.get("llvm_dis_path", toolchain / "llvm-dis"))
        if _tool_available(bundler) and _tool_available(llvm_dis):
            levels.append("instruction")
        return levels

    def capabilities(self) -> List[str]:
        return [
            "kernel_profile", "execution_timeline", "resource_usage",
            "device_binary", "device_object", "device_code", "compiler_ir",
            "source_mapping",
        ]

    def prepare(self, context: ProfileExecutionContext) -> dict[str, Any]:
        options = context.options.backend_options.get("metax", {})
        allowed = {
            "mctracer_path", "toolchain_path", "bundler_path", "llvm_dis_path",
            "llvm_nm_path", "llvm_objcopy_path",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ProfilerExecutionError(
                f"unsupported MetaX profiler options: {', '.join(unknown)}"
            )
        executable = str(options.get("mctracer_path", _DEFAULT_MCTRACER))
        if not _tool_available(executable):
            raise ProfilerUnsupportedError(
                f"mcTracer executable was not found at {executable!r}"
            )
        toolchain = Path(str(options.get("toolchain_path", _DEFAULT_TOOLCHAIN)))
        bundler = str(options.get("bundler_path", toolchain / "clang-offload-bundler"))
        llvm_dis = str(options.get("llvm_dis_path", toolchain / "llvm-dis"))
        if context.options.level == "instruction":
            missing = [name for name in (bundler, llvm_dis) if not _tool_available(name)]
            if missing:
                raise ProfilerUnsupportedError(
                    "MetaX instruction tools were not found: " + ", ".join(missing)
                )
        cache = context.artifact_dir / "triton-cache"
        cache.mkdir(exist_ok=False)
        trace_dir = context.artifact_dir / "mctracer-trace"
        trace_dir.mkdir(exist_ok=False)
        return {
            "options": options,
            "executable": executable,
            "bundler": bundler,
            "llvm_dis": llvm_dis,
            "llvm_nm": str(options.get("llvm_nm_path", toolchain / "llvm-nm")),
            "llvm_objcopy": str(
                options.get("llvm_objcopy_path", toolchain / "llvm-objcopy")
            ),
            "cache": cache,
            "trace_dir": trace_dir,
        }

    def collect(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        relative_output = os.path.relpath(
            prepared["trace_dir"], Path(context.command.cwd).resolve()
        )
        command = [
            prepared["executable"], "--mctx", "--odname", relative_output,
            "--name", "kernelgen", *context.command.argv,
        ]
        overrides = {
            "TRITON_CACHE_DIR": str(prepared["cache"]),
            "KGS_CACHE_PATH": str(context.artifact_dir / "module-cache"),
            "MCTX_TARGET_PROFILE_PATH": str(context.artifact_dir),
        }
        if context.options.level == "instruction":
            overrides["TRITON_DISABLE_LINE_INFO"] = "0"
        execution = context.run_profile_stage(
            command, label="MetaX mcTracer collection", env_overrides=overrides
        )
        log = context.artifact_dir / "mctracer.log"
        log.write_text(execution.output, encoding="utf-8")
        context.add_artifact("profile_log", "text", log, "text/plain")
        if execution.returncode != 0:
            raise ProfilerExecutionError(
                f"mcTracer exited with status {execution.returncode}"
            )
        failures = []
        for path in sorted(prepared["trace_dir"].rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"{path.name}: {exc}")
                continue
            if isinstance(payload, dict) and isinstance(payload.get("traceEvents"), list):
                context.add_artifact(
                    "vendor_report", "chrome-trace-json", path, "application/json"
                )
                return {"trace_path": path, "payload": payload}
        detail = f" ({'; '.join(failures)})" if failures else ""
        raise ProfilerExecutionError(
            f"mcTracer did not produce a readable mctx trace{detail}"
        )

    def _run_optional(
        self, context: ProfileExecutionContext, command: list[str], label: str
    ) -> Any:
        execution = context.run_tool(command, label=label)
        if execution.returncode != 0:
            context.warnings.append(
                f"{label} exited with status {execution.returncode}"
            )
            return None
        return execution

    def _unbundle(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        binary: Path,
        target: str,
        output: Path,
    ) -> bool:
        execution = self._run_optional(
            context,
            [
                prepared["bundler"], "--unbundle", "--type=o",
                f"--targets={target}", f"--input={binary}", f"--output={output}",
            ],
            f"MetaX unbundle {output.name}",
        )
        return execution is not None and output.is_file()

    def _instruction_artifacts(
        self, context: ProfileExecutionContext, prepared: dict[str, Any]
    ) -> dict[str, Any]:
        target_view = report_source_target(context.source_roots)
        binaries = sorted(prepared["cache"].rglob("*.mcfatbin"))
        operations: list[dict[str, Any]] = []
        object_dir = context.artifact_dir / "device-objects"
        object_dir.mkdir(exist_ok=False)
        object_count = bitcode_count = 0
        seen_ir: set[Path] = set()
        for index, binary in enumerate(binaries):
            context.add_artifact("device_binary", "mcfatbin", binary)
            for ir in sorted(binary.parent.iterdir()):
                if ir in seen_ir or ir.suffix not in {".ttir", ".ttgir"}:
                    continue
                seen_ir.add(ir)
                context.add_artifact("compiler_ir", ir.suffix[1:], ir, "text/plain")
                if ir.suffix == ".ttir":
                    operations.extend(
                        parse_ttir_instructions(
                            ir.read_text(encoding="utf-8", errors="replace"),
                            binary_name=binary.name,
                            target=target_view,
                        )
                    )
            listed = self._run_optional(
                context,
                [prepared["bundler"], "--list", "--type=o", f"--input={binary}"],
                f"MetaX list bundle {index}",
            )
            if listed is None:
                continue
            targets = [line.strip() for line in listed.output.splitlines() if line.strip()]
            object_target = next(
                (name for name in targets if "maca-mxc-metax" in name and not name.endswith("-bc")),
                "",
            )
            bitcode_target = next(
                (name for name in targets if "maca-mxc-metax" in name and name.endswith("-bc")),
                "",
            )
            if object_target:
                obj = object_dir / f"{index:03d}-{binary.stem}.metax.o"
                if self._unbundle(context, prepared, binary, object_target, obj):
                    object_count += 1
                    context.add_artifact("device_object", "elf64-metax", obj)
                    if _tool_available(prepared["llvm_objcopy"]):
                        raw_text = object_dir / f"{index:03d}-{binary.stem}.text.bin"
                        dumped = self._run_optional(
                            context,
                            [prepared["llvm_objcopy"], "--dump-section",
                             f".text={raw_text}", str(obj)],
                            f"MetaX extract text {index}",
                        )
                        if dumped is not None and raw_text.is_file():
                            context.add_artifact("device_code", "binary", raw_text)
                            hex_path = raw_text.with_suffix(".hex.txt")
                            hex_path.write_text(_hex_dump(raw_text.read_bytes()), encoding="utf-8")
                            context.add_artifact("device_code", "hex", hex_path, "text/plain")
            if bitcode_target:
                bitcode = object_dir / f"{index:03d}-{binary.stem}.metax.bc"
                if self._unbundle(context, prepared, binary, bitcode_target, bitcode):
                    bitcode_count += 1
                    context.add_artifact("compiler_ir", "llvm-bitcode", bitcode)
                    llvm_ir = bitcode.with_suffix(".llvm.ll")
                    disassembled = self._run_optional(
                        context,
                        [prepared["llvm_dis"], str(bitcode), "-o", str(llvm_ir)],
                        f"MetaX llvm-dis {index}",
                    )
                    if disassembled is not None and llvm_ir.is_file():
                        context.add_artifact("compiler_ir", "llvm-ir", llvm_ir, "text/plain")
        source_mapping = _source_mapping_payload(operations, target_view)
        payload = {
            "evidence_type": "static_compiler_operation_listing",
            "operation_stage": "ttir",
            "native_machine_isa": False,
            "instruction_addresses": False,
            "compiler_operation_count": len(operations),
            "mapped_compiler_operation_count": source_mapping["mapped_instruction_count"],
            "compiler_operations": operations,
        }
        listing = context.artifact_dir / "compiler-operations.json"
        listing.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        context.add_artifact(
            "compiler_operations", "json", listing, "application/json"
        )
        mapping = context.artifact_dir / "source-map.json"
        mapping.write_text(json.dumps(source_mapping, indent=2), encoding="utf-8")
        context.add_artifact("source_mapping", "json", mapping, "application/json")
        summary_text = _render_instruction_summary(
            operations,
            source_mapping,
            binary_count=len(binaries),
            object_count=object_count,
        )
        summary = context.artifact_dir / "instruction-summary.txt"
        summary.write_text(summary_text, encoding="utf-8")
        context.add_artifact("instruction_summary", "text", summary, "text/plain")
        source_report = context.artifact_dir / "source-report.txt"
        source_report.write_text(_render_source_report(source_mapping), encoding="utf-8")
        context.add_artifact("source_report", "text", source_report, "text/plain")
        context.warnings.append(
            "MetaX instruction level contains compiler operations, not native machine ISA"
        )
        return {
            "binary_count": len(binaries),
            "device_object_count": object_count,
            "device_bitcode_count": bitcode_count,
            "compiler_operation_count": len(operations),
            "mapped_compiler_operation_count": source_mapping[
                "mapped_instruction_count"
            ],
            "native_machine_isa": False,
            "instruction_addresses": False,
        }

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: dict[str, Any],
        collected: dict[str, Any],
    ) -> None:
        try:
            metrics = parse_mctracer_trace(collected["payload"])
        except ValueError as exc:
            raise ProfilerExecutionError(f"could not normalize mcTracer trace: {exc}") from exc
        timeline = context.artifact_dir / "mctracer-timeline.json"
        timeline.write_text(json.dumps(metrics["timeline"], indent=2), encoding="utf-8")
        context.add_artifact(
            "execution_timeline", "json", timeline, "application/json"
        )
        context.metrics = metrics
        context.summary.update(
            {
                "kernel_count": metrics["kernel_count"],
                "kernel_invocation_count": metrics["kernel_invocation_count"],
                "total_kernel_time_us": metrics["total_kernel_time_us"],
            }
        )
        context.add_capability("kernel_profile", "execution_timeline", "resource_usage")
        if context.options.level == "instruction":
            context.summary.update(self._instruction_artifacts(context, prepared))
            for artifact_kind in (
                "device_binary",
                "device_object",
                "device_code",
                "compiler_ir",
            ):
                if context.has_artifact(artifact_kind):
                    context.add_capability(artifact_kind)
            if context.summary.get("mapped_compiler_operation_count", 0):
                context.add_capability("source_mapping")
            else:
                context.warnings.append(
                    "MetaX TTIR operations could not be mapped to submitted source"
                )
        metrics_path = context.artifact_dir / "mctracer-metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        context.add_artifact(
            "normalized_metrics", "json", metrics_path, "application/json"
        )
        context.warnings.append(
            "mcTracer timing is diagnostic and is not evaluator latency"
        )

    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        if context.metrics.get("kernel_invocation_count", 0) < 1:
            raise ProfilerExecutionError(
                "mcTracer capture contains no device kernels"
            )
        if context.options.level == "instruction":
            if context.summary.get("binary_count", 0) < 1:
                raise ProfilerExecutionError(
                    "MetaX Triton cache did not yield a .mcfatbin"
                )
            if context.summary.get("compiler_operation_count", 0) < 1:
                raise ProfilerExecutionError(
                    "MetaX Triton cache did not yield compiler operations"
                )


__all__ = ["McTracerProfiler"]
