# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Build a private ROCtracer launcher for Hygon DTK installations."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..process import RequestDeadline


_COMPAT_VERSION = "rocm61-dtk25-v1"
_RESOURCE_DIR = Path(__file__).with_name("hygon_roctracer_compat")


class HygonTraceCompatError(RuntimeError):
    """Raised when the private Hygon ROCtracer compatibility layer cannot be prepared."""


@dataclass(frozen=True)
class HygonTraceCompat:
    launcher: Path
    library_dir: Path


def _resolve_executable(path: str) -> Path:
    resolved = shutil.which(path)
    candidate = Path(resolved or path).expanduser()
    if not candidate.is_file():
        raise HygonTraceCompatError(f"rocprof executable was not found: {path}")
    return candidate.resolve()


def _rocprof_root(path: str) -> Path:
    executable = _resolve_executable(path)
    candidates = [executable.parent.parent, executable.parent]
    candidates.extend(executable.parents)
    for root in dict.fromkeys(candidates):
        if (root / "libexec" / "rocprofiler" / "tblextr.py").is_file():
            return root
    raise HygonTraceCompatError(
        f"could not locate the rocprof installation root from {executable}"
    )


def rocprof_trace_libraries_available(path: str) -> bool:
    """Return whether a rocprof installation already contains its trace toolchain."""
    try:
        root = _rocprof_root(path)
    except HygonTraceCompatError:
        return False
    tracer_runtime = (
        root / "lib" / "libroctracer64.so.4"
        if (root / "lib" / "libroctracer64.so.4").is_file()
        else root.parent / "roctracer" / "lib" / "libroctracer64.so.4"
    )
    return all(
        candidate.is_file()
        for candidate in (
            tracer_runtime,
            root / "lib" / "roctracer" / "libroctracer_tool.so",
            root / "lib" / "roctracer" / "libfile_plugin.so",
        )
    )


CommandRunner = Callable[[Sequence[str], str], tuple[int, str]]


def _run(
    command: Sequence[str],
    *,
    label: str,
    command_runner: CommandRunner | None = None,
) -> None:
    if command_runner is None:
        process = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        returncode = process.returncode
        output = process.stdout or ""
    else:
        returncode, output = command_runner(command, label)
    if returncode == 0:
        return
    output = output.strip()
    if len(output) > 4_000:
        output = output[-4_000:]
    detail = f":\n{output}" if output else ""
    raise HygonTraceCompatError(f"{label} failed with status {returncode}{detail}")


def _include_flags(dtk_root: Path) -> list[str]:
    required_include_dirs = (
        _RESOURCE_DIR / "src" / "tracer_tool",
        _RESOURCE_DIR / "src" / "util",
        _RESOURCE_DIR / "include",
        dtk_root / "hip" / "include",
        dtk_root / ".hyhal" / "include",
        dtk_root / "include",
    )
    missing = [str(path) for path in required_include_dirs if not path.is_dir()]
    if missing:
        raise HygonTraceCompatError(
            "DTK ROCtracer development headers are incomplete; missing " + ", ".join(missing)
        )
    roctracer_candidates = (
        dtk_root / "roctracer" / "include",
        dtk_root / "hip" / "include" / "roctracer",
    )
    roctracer_include_dirs = tuple(path for path in roctracer_candidates if path.is_dir())
    if not roctracer_include_dirs:
        raise HygonTraceCompatError(
            "DTK ROCtracer development headers are incomplete; expected one of "
            + ", ".join(str(path) for path in roctracer_candidates)
        )
    include_dirs = (*required_include_dirs, *roctracer_include_dirs)
    return [flag for path in include_dirs for flag in ("-I", str(path))]


def _build_libraries(
    build_dir: Path,
    dtk_root: Path,
    *,
    command_runner: CommandRunner | None = None,
) -> None:
    compiler = shutil.which("g++")
    if not compiler:
        raise HygonTraceCompatError("g++ is required to build the Hygon ROCtracer compatibility layer")

    hip_lib = dtk_root / "hip" / "lib"
    hsa_lib = dtk_root / ".hyhal" / "lib"
    dtk_lib = dtk_root / "lib"
    galaxyhip = hip_lib / "libgalaxyhip.so"
    required_libraries = (
        galaxyhip,
        hsa_lib / "libhsa-runtime64.so",
        dtk_lib / "libamd_comgr.so",
    )
    missing = [str(path) for path in required_libraries if not path.exists()]
    if missing:
        raise HygonTraceCompatError(
            "DTK runtime libraries are incomplete; missing " + ", ".join(missing)
        )

    common = [
        compiler,
        "-std=c++17",
        "-fPIC",
        "-O2",
        "-DAMD_INTERNAL_BUILD",
        "-DHIP_PROF_HIP_API_STRING=1",
        "-D__HIP_PLATFORM_AMD__=1",
        *_include_flags(dtk_root),
    ]
    sources = {
        "debug.o": _RESOURCE_DIR / "src" / "util" / "debug.cpp",
        "util.o": _RESOURCE_DIR / "src" / "util" / "util.cpp",
        "tracer_tool.o": _RESOURCE_DIR / "src" / "tracer_tool" / "tracer_tool.cpp",
        "file_plugin.o": _RESOURCE_DIR / "plugin" / "file" / "file.cpp",
    }
    for output_name, source in sources.items():
        _run(
            [*common, "-c", str(source), "-o", str(build_dir / output_name)],
            label=f"compiling {source.name}",
            command_runner=command_runner,
        )

    rpath = f"{hip_lib}:{hsa_lib}:{dtk_lib}"
    _run(
        [
            compiler,
            "-shared",
            f"-Wl,--version-script={_RESOURCE_DIR / 'src' / 'tracer_tool' / 'exportmap'}",
            "-Wl,--no-undefined",
            f"-Wl,-rpath,{rpath}",
            "-o",
            str(build_dir / "libroctracer_tool.so"),
            str(build_dir / "tracer_tool.o"),
            str(build_dir / "debug.o"),
            str(build_dir / "util.o"),
            "-L",
            str(hip_lib),
            "-lgalaxyhip",
            "-L",
            str(hsa_lib),
            "-lhsa-runtime64",
            "-lstdc++fs",
            "-latomic",
            "-ldl",
            "-lpthread",
        ],
        label="linking libroctracer_tool.so",
        command_runner=command_runner,
    )
    _run(
        [
            compiler,
            "-shared",
            f"-Wl,--version-script={_RESOURCE_DIR / 'plugin' / 'exportmap'}",
            "-Wl,--no-undefined",
            f"-Wl,-rpath,{rpath}",
            "-o",
            str(build_dir / "libfile_plugin.so"),
            str(build_dir / "file_plugin.o"),
            str(build_dir / "debug.o"),
            str(build_dir / "util.o"),
            "-L",
            str(hip_lib),
            "-lgalaxyhip",
            "-L",
            str(hsa_lib),
            "-lhsa-runtime64",
            "-L",
            str(dtk_lib),
            "-lamd_comgr",
            "-lstdc++fs",
            "-ldl",
            "-lpthread",
        ],
        label="linking libfile_plugin.so",
        command_runner=command_runner,
    )

    tracer_alias = build_dir / "libroctracer64.so.4"
    if os.path.lexists(tracer_alias):
        tracer_alias.unlink()
    tracer_alias.symlink_to(galaxyhip)


def _render_launcher(
    build_dir: Path,
    rocprof_root: Path,
    *,
    published_dir: Path | None = None,
) -> Path:
    template_path = _RESOURCE_DIR / "rocprof.in"
    if not template_path.is_file():
        raise HygonTraceCompatError(f"rocprof compatibility template is missing: {template_path}")
    launcher = build_dir / "rocprof-hygon"
    text = template_path.read_text(encoding="utf-8")
    text = text.replace("__ROCPROF_ROOT__", shlex.quote(str(rocprof_root)))
    text = text.replace(
        "__COMPAT_LIB_DIR__",
        shlex.quote(str(published_dir or build_dir)),
    )
    launcher.write_text(text, encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def _ready(build_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            build_dir / "rocprof-hygon",
            build_dir / "libroctracer_tool.so",
            build_dir / "libfile_plugin.so",
        )
    ) and os.path.lexists(build_dir / "libroctracer64.so.4")


def _dtk_cache_identity(dtk_root: Path, rocprof_root: Path) -> str:
    """Fingerprint the installation identity without importing its runtime."""

    digest = hashlib.sha256()
    digest.update(_COMPAT_VERSION.encode("utf-8"))
    for path in (
        dtk_root,
        rocprof_root / "bin" / "rocprof",
        dtk_root / "hip" / "lib" / "libgalaxyhip.so",
        dtk_root / ".hyhal" / "lib" / "libhsa-runtime64.so",
        dtk_root / "lib" / "libamd_comgr.so",
    ):
        resolved = path.resolve()
        digest.update(str(resolved).encode("utf-8"))
        try:
            stat = resolved.stat()
        except OSError:
            digest.update(b"\0missing")
        else:
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()[:16]


def prepare_hygon_rocprof(
    path: str,
    cache_dir: str | Path | None = None,
    *,
    command_runner: CommandRunner | None = None,
    deadline: RequestDeadline | None = None,
) -> HygonTraceCompat:
    """Build or reuse a private rocprof trace runtime without modifying the DTK install."""
    rocprof_root = _rocprof_root(path)
    dtk_root = rocprof_root.parent.resolve()
    cache_root = Path(
        cache_dir
        or os.environ.get("KERNELGEN_HYGON_TRACE_CACHE")
        or (Path(tempfile.gettempdir()) / "kernelgen-hygon-roctracer")
    ).expanduser()
    safe_dtk_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", dtk_root.name)
    identity = _dtk_cache_identity(dtk_root, rocprof_root)
    build_dir = cache_root / f"{_COMPAT_VERSION}-{safe_dtk_name}-{identity}"
    cache_root.mkdir(parents=True, exist_ok=True)

    lock_path = cache_root / f".{build_dir.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is None:
                    time.sleep(0.05)
                else:
                    time.sleep(
                        min(
                            0.05,
                            deadline.remaining(
                                "waiting for Hygon ROCtracer compatibility lock"
                            ),
                        )
                    )
        if not _ready(build_dir):
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{build_dir.name}.build-",
                    dir=cache_root,
                )
            )
            try:
                _build_libraries(
                    temporary,
                    dtk_root,
                    command_runner=command_runner,
                )
                _render_launcher(
                    temporary,
                    rocprof_root,
                    published_dir=build_dir,
                )
                if not _ready(temporary):
                    raise HygonTraceCompatError(
                        "Hygon ROCtracer compatibility build is incomplete"
                    )
                if build_dir.is_symlink():
                    raise HygonTraceCompatError(
                        f"refusing to replace symlinked compatibility cache {build_dir}"
                    )
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                temporary.replace(build_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)

    if not _ready(build_dir):
        raise HygonTraceCompatError(
            f"Hygon ROCtracer compatibility build is incomplete under {build_dir}"
        )
    return HygonTraceCompat(build_dir / "rocprof-hygon", build_dir)
