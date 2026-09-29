# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Device abstraction for the KernelGen Server runtime.

Collects all backend-specific operations (device sync, memory cleanup, device
discovery, and — most importantly — kernel timing) behind a single ``Device``
interface, so the evaluator/runner/timing code stays backend-agnostic.

Design constraints:

- **Zero vendor import side effects.** Heavy backend-only imports such as
  ``torch_npu`` are done *lazily* inside
  the concrete ``Device`` methods, never at module load. This is what lets an
  NPU host import the engine without pulling in CUDA-only libraries.
- **``device`` flows as a bare runtime string** ("cuda:0", "npu:0", "gcu:0").
  CUDA-compatible vendor backends use ``cuda:<index>``; Enflame uses ``gcu``.
  The configured logical backend selects the matching vendor implementation.
- **Per-backend implementation lives in its own package:**
  ``kernelgen_server/backends/<vendor>/``. Adding a new chip = add a new
  package with a ``Device`` subclass.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time as _time_module
import xml.etree.ElementTree as ElementTree
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

# NOTE: torch is safe to import at module load on both CUDA and NPU hosts
# (torch_npu is a *separate* import). Backend-only libs are imported lazily.
import torch

from .backend import (
    CUDA_COMPATIBLE_BACKENDS,
    KNOWN_BACKENDS,
    runtime_device_type,
)

DEVICE_BACKEND_ENV = "KGS_DEVICE_BACKEND"
TIMING_STRATEGY_ENV = "KGS_TIMING_STRATEGY"
_UNAVAILABLE_STATUS_VALUES = frozenset(
    {
        "n/a",
        "na",
        "none",
        "not available",
        "not found",
        "not supported",
        "null",
        "unknown",
        "unknown error",
        "unsupported",
    }
)


def clean_status_value(value: object) -> str:
    """Normalize a probed value without manufacturing missing metadata."""

    if value is None:
        return ""
    text = str(value).strip()
    normalized = re.sub(r"\s+", " ", text.casefold()).strip(" :-")
    if not normalized or normalized in _UNAVAILABLE_STATUS_VALUES:
        return ""
    if normalized.startswith(("n/a ", "not available ", "not found ")):
        return ""
    return text


def run_status_command(argv: List[str], timeout: float = 10.0) -> str:
    """Run one read-only vendor query without a shell."""

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return "\n".join(
        part for part in (result.stdout, result.stderr) if part
    ).strip()


def status_value_from_label(output: str, *labels: str) -> str:
    """Extract the first available ``Label: value`` from command output."""

    for label in labels:
        match = re.search(
            rf"(?im)^\s*{re.escape(label)}\s*[:=]\s*(.*?)\s*$",
            output,
        )
        if match is not None:
            value = clean_status_value(match.group(1))
            if value:
                return value
    return ""


def status_json_from_output(output: str) -> object:
    """Decode JSON even when a vendor CLI prints a banner first."""

    starts = sorted(
        index for index in (output.find("{"), output.find("[")) if index >= 0
    )
    for start in starts:
        try:
            value, _ = json.JSONDecoder().raw_decode(output[start:])
        except (json.JSONDecodeError, TypeError):
            continue
        return value
    return None


def status_json_value(payload: object, *keys: str) -> str:
    """Recursively find one named scalar in a vendor JSON payload."""

    wanted = {
        re.sub(r"[-_\s]+", "", key).casefold()
        for key in keys
    }

    def visit(value: object) -> str:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = re.sub(r"[-_\s]+", "", str(key)).casefold()
                if normalized in wanted and not isinstance(child, (dict, list)):
                    found = clean_status_value(child)
                    if found:
                        return found
            for child in value.values():
                found = visit(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = visit(child)
                if found:
                    return found
        return ""

    return visit(payload)


def status_xml_value(output: str, *tags: str) -> str:
    """Read the first available scalar from vendor XML output."""

    start = output.find("<")
    if start < 0:
        return ""
    try:
        root = ElementTree.fromstring(output[start:])
    except ElementTree.ParseError:
        return ""
    wanted = {tag.casefold() for tag in tags}
    for element in root.iter():
        local_name = str(element.tag).rsplit("}", 1)[-1].casefold()
        if local_name in wanted:
            value = clean_status_value(element.text)
            if value:
                return value
    return ""


def torch_runtime_version(name: str) -> str:
    """Read a runtime version explicitly exported by installed PyTorch."""

    return clean_status_value(getattr(getattr(torch, "version", None), name, ""))


def _property_value(properties: object, *names: str) -> object:
    for name in names:
        value = getattr(properties, name, None)
        if value not in (None, ""):
            return value
    return None


def _property_architecture(properties: object) -> str:
    """Read architecture from runtime properties, never from a model map."""

    model = clean_status_value(_property_value(properties, "name"))
    model_key = re.sub(r"[^a-z0-9]+", "", model.casefold())
    for attribute in ("gcnArchName", "gcn_arch_name", "arch_name"):
        candidate = clean_status_value(_property_value(properties, attribute))
        candidate_key = re.sub(r"[^a-z0-9]+", "", candidate.casefold())
        if (
            candidate
            and candidate_key not in {"device", "gcu", "gpu"}
            and candidate_key != model_key
        ):
            return candidate
    major = _property_value(properties, "major")
    minor = _property_value(properties, "minor")
    if major is not None and minor is not None:
        return f"{major}.{minor}"
    return ""


def device_info_from_properties(
    device_id: int,
    name: str,
    properties: object,
    *,
    architecture: str = "",
    num_sm: int = 0,
) -> "DeviceInfo":
    """Convert vendor runtime properties to the common best-effort schema."""

    return DeviceInfo(
        device_id=device_id,
        name=clean_status_value(name)
        or clean_status_value(_property_value(properties, "name")),
        l2_cache_size=int(
            _property_value(properties, "L2_cache_size", "l2_cache_size") or 0
        ),
        num_sm=int(
            num_sm
            or _property_value(
                properties,
                "multi_processor_count",
                "multiProcessorCount",
            )
            or 0
        ),
        compute_capability=clean_status_value(architecture)
        or _property_architecture(properties),
        total_memory=int(_property_value(properties, "total_memory") or 0),
    )


# ---------------------------------------------------------------------------
# DeviceInfo — hardware properties for roofline / autotune / profiling hints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware properties of a specific device, used for roofline analysis and
    autotune hints. Fields are best-effort — backends that can't query a field
    should leave it at the default (0 / empty).
    """

    device_id: int = 0
    name: str = ""
    l2_cache_size: int = 0  # bytes
    num_sm: int = 0  # SM / AI Core count
    compute_capability: str = ""  # e.g. "8.0" for A100, "" if unavailable
    total_memory: int = 0  # bytes


class Device(ABC):
    """Backend-specific device operations used by the evaluation engine."""

    #: Logical service backend name, e.g. "cuda", "npu", or "iluvatar".
    backend: str = ""

    # -- device addressing ------------------------------------------------

    def parse_index(self, device: str) -> int:
        """Parse the integer index out of a ``"<backend>:<idx>"`` string."""
        if ":" in device:
            return int(device.split(":")[1])
        return 0

    @staticmethod
    def visible_device_count(env_var: str) -> Optional[int]:
        """Count an explicit visibility list, or return ``None`` when unset."""
        value = os.environ.get(env_var)
        if value is None:
            return None
        entries = [item.strip() for item in value.split(",") if item.strip()]
        if not entries or any(item == "-1" for item in entries):
            return 0
        return len(entries)

    @classmethod
    def constrain_visible_device_count(cls, detected: int, env_var: str) -> int:
        """Apply a visibility list after a vendor has been positively detected."""
        visible = cls.visible_device_count(env_var)
        return detected if visible is None else min(detected, visible)

    @abstractmethod
    def set_device(self, device: str) -> None:
        """Make ``device`` the active device for the current process."""

    @abstractmethod
    def synchronize(self, device: str) -> None:
        """Block until all work on ``device`` has completed."""

    @abstractmethod
    def empty_cache(self) -> None:
        """Release cached device memory back to the allocator/OS."""

    # -- discovery / naming ----------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is usable in the current process."""

    @abstractmethod
    def list_devices(self) -> List[str]:
        """List device strings, e.g. ``["cuda:0", "cuda:1"]``."""

    @abstractmethod
    def device_name(self, device: str) -> str:
        """Human-readable hardware name for ``device``."""

    def count_devices_safe(self) -> int:
        """Count available devices WITHOUT importing device-specific libraries.

        Safe to call from the server main process (which must not import
        torch_npu / kernelgen_server). Default uses subprocess; backends can override
        with lighter methods (e.g. parsing CUDA_VISIBLE_DEVICES).
        """
        import subprocess as _sp

        try:
            result = _sp.run(
                [sys.executable, "-c", f"import torch; print(torch.{self.backend}.device_count())"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception:
            return 0

    # -- per-backend metadata (override in subclasses) --------------------

    @property
    def default_target_hardware(self) -> List[str]:
        """Default target_hardware for Solutions on this backend."""
        return [self.backend.upper()]

    @property
    def default_entry_point(self) -> str:
        """Default entry_point for Solutions on this backend."""
        return "main.py::run"

    # -- timing -----------------------------------------------------------

    def time(
        self,
        fn: Callable,
        args: List[Any],
        warmup: int,
        iters: int,
        device: str,
        grad_to_none: Optional[List[Any]] = None,
    ) -> float:
        """Time ``fn(*args)`` on ``device``.

        Returns the representative latency in **milliseconds**.

        ``warmup`` and ``iters`` are **time budgets in milliseconds**, matching
        ``triton.testing.do_bench`` (and FlagGems' ``warm_up`` / ``repetition``),
        NOT iteration counts. Each backend either hands the budget straight to
        ``do_bench`` (which probes per-call latency and derives counts) or converts
        it to counts itself via :meth:`budget_to_counts`. Keeping one unit at this
        boundary makes the CUDA and NPU paths sample comparably instead of the two
        silently interpreting the same numbers differently.

        Default implementation uses ``triton.testing.do_bench`` — available in
        every vendor's Triton fork (NVIDIA/Ascend/Moore/MetaX/...). Handles L2
        cache clearing and returns the **median**. Backends can override with
        more precise methods (e.g. CUPTI for NVIDIA, profiler_npu for Ascend).

        ``grad_to_none`` (optional) is forwarded to ``do_bench`` so backward-pass
        kernels have their input gradients cleared between iterations, preventing
        gradient accumulation from distorting the measured latency.

        Implementing a new backend
        --------------------------
        You do NOT need to reimplement :meth:`budget_to_counts` — it is defined on
        this base class and inherited unchanged (it relies only on ``synchronize`` +
        a wall-clock probe, no chip-specific logic). Two cases:

        * If your backend can use the default ``do_bench`` path, do not override
          ``time`` at all — the ms budget flows straight through.
        * If you override ``time`` and your underlying timing API takes iteration
          **counts** (e.g. CUPTI ``repeat_iters``, a profiler ``active`` window, or a
          hand-rolled ``for`` loop), you MUST first convert the budget with
          ``warmup_n, rep_n = self.budget_to_counts(fn, args, warmup, iters, device)``.
          Do NOT pass the ms budget where a count is expected — feeding e.g.
          ``iters`` (100 ms) directly into a 100-iteration loop silently under- or
          over-samples and desyncs this backend from every other one.
        """
        import triton.testing

        latency_ms = float(triton.testing.do_bench(
            lambda: fn(*args),
            warmup=warmup,
            rep=iters,
            return_mode="median",
            grad_to_none=grad_to_none,
        ))
        if not latency_ms > 0:
            raise RuntimeError(f"do_bench returned invalid latency: {latency_ms!r}")
        return latency_ms

    def get_device_info(self, device: str) -> DeviceInfo:
        """Query hardware properties for a specific device.

        CUDA-compatible vendor runtimes expose their properties through
        ``torch.cuda``. Unsupported backends retain a name-only fallback.
        """
        index = self.parse_index(device)
        if self.backend in CUDA_COMPATIBLE_BACKENDS:
            try:
                properties = torch.cuda.get_device_properties(index)
                return device_info_from_properties(
                    index,
                    self.device_name(device),
                    properties,
                )
            except Exception:
                pass
        return DeviceInfo(device_id=index, name=self.device_name(device))

    def status_metadata(self) -> dict[str, object]:
        """Return dynamically observed architecture/runtime/driver metadata."""

        return {
            "architecture": "",
            "runtime_version": "",
            "driver_version": "",
            "sources": {},
        }

    # -- auto-calibrated timing helper ---------------------------------------

    def auto_calibrate_iters(
        self,
        fn: Callable,
        args: List[Any],
        device: str,
        warmup_budget_ms: float = 1000.0,
        rep_budget_ms: float = 100.0,
        probe_runs: int = 5,
    ) -> tuple:
        """Estimate (warmup_iters, timing_iters) based on time budgets.

        Runs ``fn`` a few times to estimate per-call latency, then computes how
        many iterations fit into the warmup and repetition time budgets. This
        adapts to fast kernels (many iters for precision) and slow kernels (few
        iters to avoid wasting time).

        Inspired by FlagGems ``get_iter_count()``.

        Returns
        -------
        (warmup_iters, timing_iters) : tuple[int, int]
        """
        # Probe: run a few times and measure wall-clock
        for _ in range(2):
            fn(*args)
        self.synchronize(device)

        t0 = _time_module.perf_counter()
        for _ in range(probe_runs):
            fn(*args)
        self.synchronize(device)
        elapsed_ms = (_time_module.perf_counter() - t0) * 1000.0

        latency_ms = elapsed_ms / probe_runs if probe_runs > 0 else 1.0
        latency_ms = max(latency_ms, 0.001)  # guard against zero

        warmup_iters = max(1, int(warmup_budget_ms / latency_ms))
        timing_iters = max(1, int(rep_budget_ms / latency_ms))
        return warmup_iters, timing_iters

    def budget_to_counts(
        self,
        fn: Callable,
        args: List[Any],
        warmup_ms: float,
        rep_ms: float,
        device: str,
        max_warmup: int = 100_000,
        max_rep: int = 100_000,
    ) -> tuple:
        """Convert ms time budgets to (warmup_count, rep_count) by probing latency.

        This is the count-based analogue of what ``triton.testing.do_bench`` does
        internally: run ``fn`` a few times to estimate per-call latency, then divide
        the budgets by it. Used by backends whose timing loops take iteration counts
        (NPU walltime / profiler) so they honor the same ms budget the ``do_bench``
        path does. Mirrors FlagGems ``get_iter_count()``.

        Counts are clamped to sane ceilings so a pathologically fast probe can't ask
        for an unbounded loop.
        """
        w, r = self.auto_calibrate_iters(
            fn, args, device, warmup_budget_ms=warmup_ms, rep_budget_ms=rep_ms
        )
        return min(w, max_warmup), min(r, max_rep)


# ---------------------------------------------------------------------------
# Registry + selection
# ---------------------------------------------------------------------------

# Per-backend singletons (cached so timing state / config is stable per process).
_INSTANCES: dict[str, Device] = {}

# Optional overrides applied when a backend is first instantiated, e.g.
# ``configure_device("npu", perf_mode="walltime")`` set by the server at startup.
_CONFIG: dict[str, dict] = {}


def configure_device(backend: str, **kwargs) -> None:
    """Set construction kwargs for a backend before first use (e.g. NPU perf_mode).

    For config that must survive a process spawn (the isolated runner spawns a
    fresh worker that re-imports this module), also export it via env vars so the
    subprocess picks it up.
    """
    if backend not in KNOWN_BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}")
    device_kwargs = dict(kwargs)
    timing_strategy = device_kwargs.pop("timing_strategy", None)
    if (
        timing_strategy is not None
        and timing_strategy not in {"triton", "profiler", "walltime"}
    ):
        raise ValueError(f"Unknown timing strategy: {timing_strategy!r}")

    if backend == "kunlunxin":
        for name in (
            "TORCHINDUCTOR_COMPILE_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            os.environ.setdefault(name, "1")
    _CONFIG[backend] = device_kwargs
    _INSTANCES.pop(backend, None)  # force re-instantiation with new config
    # The isolated runner uses multiprocessing spawn, so the logical backend
    # must also cross the process boundary. This lets ``get_device("cuda:0")``
    # resolve to the configured CUDA-compatible vendor implementation.
    os.environ[DEVICE_BACKEND_ENV] = backend
    if timing_strategy is not None:
        os.environ[TIMING_STRATEGY_ENV] = timing_strategy
    if backend == "npu" and "perf_mode" in device_kwargs:
        from kernelgen_server.backends.ascend import NpuDevice

        os.environ[NpuDevice.PERF_MODE_ENV] = device_kwargs["perf_mode"]


def _make_device(backend: str) -> Device:
    cfg = _CONFIG.get(backend, {})
    if backend == "cuda":
        from kernelgen_server.backends.nvidia import CudaDevice

        return CudaDevice()
    if backend == "npu":
        from kernelgen_server.backends.ascend import NpuDevice

        return NpuDevice(**cfg)
    if backend == "musa":
        from kernelgen_server.backends.mthreads import MusaDevice

        return MusaDevice()
    if backend == "mlu":
        from kernelgen_server.backends.cambricon import CambriconDevice

        return CambriconDevice()
    if backend == "txda":
        from kernelgen_server.backends.tsingmicro import TxdaDevice

        return TxdaDevice()
    # Vendor runtimes exposed through torch.cuda use "cuda" device strings, but
    # keep a distinct logical backend for discovery, policy, and profiling.
    if backend == "hygon":
        from kernelgen_server.backends.hygon import HygonDevice

        return HygonDevice()
    if backend == "metax":
        from kernelgen_server.backends.metax import MetaxDevice

        return MetaxDevice()
    if backend == "iluvatar":
        from kernelgen_server.backends.iluvatar import IluvatarDevice

        return IluvatarDevice()
    if backend == "kunlunxin":
        from kernelgen_server.backends.kunlunxin import KunlunxinDevice

        return KunlunxinDevice()
    if backend == "thead":
        from kernelgen_server.backends.thead import TheadDevice

        return TheadDevice()
    if backend == "enflame":
        from kernelgen_server.backends.enflame import EnflameDevice

        return EnflameDevice()
    raise ValueError(f"Unknown backend: {backend!r}")


def detect_backend() -> str:
    """Detect the available backend on this host (npu preferred if present)."""
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    try:
        import torch_txda  # noqa: F401

        if torch.txda.is_available():
            return "txda"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("No supported device backend (cuda / npu / txda) detected")


def _backend_of(device_or_backend: Optional[str]) -> str:
    s = detect_backend() if device_or_backend is None else device_or_backend
    if ":" in s:
        s = s.split(":")[0]
    configured = os.environ.get(DEVICE_BACKEND_ENV, "")
    if s == "gcu" and configured == "enflame":
        return "enflame"
    if s in KNOWN_BACKENDS:
        if (
            configured in KNOWN_BACKENDS
            and configured != s
            and runtime_device_type(configured) == s
        ):
            return configured
        return s
    # Not a known backend token — treat as a request to auto-detect.
    return detect_backend()


def get_device(device_or_backend: Optional[str] = None) -> Device:
    """Return the cached ``Device`` for a device string / backend / auto-detect.

    Examples::

        get_device("cuda:0")   # -> CudaDevice
        get_device("npu:1")    # -> NpuDevice
        get_device("npu")      # -> NpuDevice
        get_device()           # -> auto-detected backend
    """
    backend = _backend_of(device_or_backend)
    dev = _INSTANCES.get(backend)
    if dev is None:
        dev = _make_device(backend)
        _INSTANCES[backend] = dev
    return dev
