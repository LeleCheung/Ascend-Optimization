# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility for Enflame FlagTree/GCU line-info toolchain skew.

Some Enflame images pair a FlagTree backend that requests
``--ensure-debug-info-scope-on-llvm-func`` with a newer ``triton-gcu`` package
whose ``gcu-compiler-opt`` no longer registers that pass.  The FlagTree opt
still owns the line-info pass, while the system opt owns the GCU lowering
passes.  For instruction profiling, replay the FlagTree lowering with debug
printing, run the line-info pass after the system lowering, and suppress the
duplicate legacy pass in the final fatbin driver.

The patch is process-local and is installed only when line info was explicitly
enabled with ``TRITON_DISABLE_LINE_INFO=0``.  It never edits the vendor package
or ``/opt/triton_gcu``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_LEGACY_LINE_INFO_PASS = "--ensure-debug-info-scope-on-llvm-func"
_CURRENT_LINE_INFO_PASS = "--enable-line-info"
_LINE_INFO_PASSES = (_LEGACY_LINE_INFO_PASS, _CURRENT_LINE_INFO_PASS)
_PATCH_MARKER = "_kernelgen_line_info_compat_installed"


class EnflameLineInfoCompatibilityError(RuntimeError):
    """Raised when the installed FlagTree/GCU tools cannot preserve line info."""


def _line_info_enabled(env: Mapping[str, str] = os.environ) -> bool:
    return env.get("TRITON_DISABLE_LINE_INFO", "1").strip().lower() in {
        "0",
        "false",
        "off",
    }


@lru_cache(maxsize=None)
def _tool_help(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout or ""


def _tool_supports_pass(executable: Path, pass_name: str) -> bool:
    return pass_name in _tool_help(str(executable))


def _select_line_info_pass(executable: Path) -> str:
    for pass_name in _LINE_INFO_PASSES:
        if _tool_supports_pass(executable, pass_name):
            return pass_name
    raise EnflameLineInfoCompatibilityError(
        f"{executable} does not provide a supported LLVM line-info pass"
    )


def _format_pass_arg(name: str, options: str = "") -> str:
    argument = name if name.startswith("-") else f"-{name}"
    return f"{argument}={options}" if options else argument


def _extract_arch(module_path: str) -> str:
    match = re.search(r"gcu(?:triton_)?(\d+)", module_path)
    return f"gcu{match.group(1)}" if match else "gcu300"


def _restore_overflow_flags(text: str) -> str:
    values = {"0": "none", "1": "nsw", "2": "nuw", "3": "nsw, nuw"}
    return re.sub(
        r"overflowFlags\s*=\s*(\d+)\s*:\s*i32",
        lambda match: (
            f"overflowFlags = #llvm.overflow<{values.get(match.group(1), 'none')}>"
        ),
        text,
    )


_TARGETS_PATTERN = re.compile(r",\s*targets\s*=\s*\[(#gcu\.target[^\]\n]*)\]")


def _detach_gcu_target(llir: str) -> tuple[str, str]:
    match = _TARGETS_PATTERN.search(llir)
    if match is None:
        raise EnflameLineInfoCompatibilityError(
            "Lowered Enflame LLIR did not contain a gcu.target attribute"
        )
    return llir[: match.start()] + llir[match.end() :], match.group(1)


def _reattach_gcu_target(llir: str, target: str) -> str:
    marker = '<{sym_name = "triton"}>'
    if marker not in llir:
        raise EnflameLineInfoCompatibilityError(
            "FlagTree line-info output did not contain the triton gpu.module"
        )
    return llir.replace(
        marker,
        f'<{{sym_name = "triton", targets = [{target}]}}>',
        1,
    )


def _triton_opt_path(toolkit: Any, arch: str) -> Path:
    configured_root = getattr(toolkit, "PY_TOOLS_PATH", None)
    root = Path(configured_root) if configured_root else Path(toolkit.__file__).parent
    return root / f"triton-{arch}-opt"


def _run_triton_opt(
    toolkit: Any,
    arch: str,
    input_ir: str,
    pass_args: Sequence[str],
) -> str:
    executable = _triton_opt_path(toolkit, arch)
    command = [
        str(executable),
        "-mlir-print-op-generic",
        "--mlir-print-debuginfo",
        *pass_args,
    ]
    try:
        result = subprocess.run(
            command,
            input=input_ir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise EnflameLineInfoCompatibilityError(
            f"Could not execute FlagTree line-info tool {executable}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise EnflameLineInfoCompatibilityError(
            f"FlagTree line-info lowering failed with {executable}: {detail}"
        )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    return result.stdout


def _run_toolkit_command(
    executable: Path,
    content: Any,
    args: Iterable[str],
    env: dict[str, str],
) -> str:
    try:
        result = subprocess.run(
            [str(executable), *args],
            input=str(content),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )
    except OSError as exc:
        raise EnflameLineInfoCompatibilityError(
            f"Could not execute GCU compiler driver {executable}: {exc}"
        ) from exc
    if result.returncode != 0:
        raise EnflameLineInfoCompatibilityError(result.stderr.strip())
    if result.stderr:
        print(
            result.stderr,
            file=sys.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
        )
    return result.stdout


def _install_compatibility(toolkit: Any) -> None:
    original_pipeline_add_pass = toolkit.Pipeline.add_pass
    original_gcu_compiler_opt = toolkit.gcu_compiler_opt

    def record_pipeline_pass(self: Any, name: str, options: str = "") -> Any:
        recorded = getattr(self, "_kernelgen_line_info_passes", None)
        if recorded is None:
            recorded = []
            self._kernelgen_line_info_passes = recorded
        recorded.append((name, options))
        return original_pipeline_add_pass(self, name, options)

    def run_pipeline_with_locations(self: Any, input_ir: str) -> str:
        pass_args = [
            _format_pass_arg(name, options)
            for name, options in getattr(self, "_kernelgen_line_info_passes", ())
            if name != "mlir-print-op-generic"
        ]
        arch = _extract_arch(str(self._mod.__file__))
        return _run_triton_opt(toolkit, arch, input_ir, pass_args)

    def gcu_compiler_opt_with_line_info(content: Any, *args: str) -> str:
        requested = next((item for item in args if item in _LINE_INFO_PASSES), "")
        if not requested:
            return original_gcu_compiler_opt(content, *args)

        filtered = [item for item in args if item not in _LINE_INFO_PASSES]
        filtered.insert(0, "-mlir-print-debuginfo")
        llir = original_gcu_compiler_opt(content, *filtered)

        arch = "gcu300"
        for item in filtered:
            if item.startswith("-insert-local-fence=arch="):
                arch = item.rsplit("=", 1)[-1]
                break

        sanitized, target = _detach_gcu_target(llir)
        triton_opt = _triton_opt_path(toolkit, arch)
        line_info_pass = _select_line_info_pass(triton_opt)
        debug_llir = _run_triton_opt(
            toolkit,
            arch,
            sanitized,
            ["--allow-unregistered-dialect", line_info_pass],
        )
        return _restore_overflow_flags(_reattach_gcu_target(debug_llir, target))

    def compile_without_duplicate_line_info_pass(content: Any, *args: str) -> str:
        child_env = os.environ.copy()
        child_env["TRITON_DISABLE_LINE_INFO"] = "1"
        executable = Path(toolkit.TOOLKIT_PATH) / "gcu-compiler-compile"
        return _run_toolkit_command(executable, content, args, child_env)

    toolkit.Pipeline.add_pass = record_pipeline_pass
    toolkit.Pipeline.run = run_pipeline_with_locations
    toolkit.gcu_compiler_opt = gcu_compiler_opt_with_line_info
    toolkit.compile = compile_without_duplicate_line_info_pass
    setattr(toolkit, _PATCH_MARKER, True)


def enable_enflame_line_info_compat() -> bool:
    """Install the process-local compatibility patch when the image needs it.

    Returns ``True`` when the compatibility path is active.  Returns ``False``
    when line info was not requested or the system compiler already supports
    the pass natively.
    """

    if not _line_info_enabled():
        return False

    try:
        from triton.backends.enflame import toolkit
    except (ImportError, ModuleNotFoundError) as exc:
        raise EnflameLineInfoCompatibilityError(
            "TRITON_DISABLE_LINE_INFO=0 was requested, but the Enflame "
            "FlagTree backend is unavailable"
        ) from exc

    if getattr(toolkit, _PATCH_MARKER, False):
        return True

    gcu_opt = Path(toolkit.TOOLKIT_PATH) / "gcu-compiler-opt"
    if _tool_supports_pass(gcu_opt, _LEGACY_LINE_INFO_PASS):
        return False

    _install_compatibility(toolkit)
    return True


__all__ = [
    "EnflameLineInfoCompatibilityError",
    "enable_enflame_line_info_compat",
]
