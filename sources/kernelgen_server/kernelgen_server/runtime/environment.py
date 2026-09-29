"""Environment helpers that avoid vendor accelerator imports in the server."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from typing import MutableMapping, Sequence

from kernelgen_client.device_visibility import (
    ALL_VISIBILITY_VARIABLES,
    BACKEND_VISIBILITY_VARIABLES as _BACKEND_VISIBILITY_VARIABLES,
)

_VENDORS = {
    "cuda": "nvidia",
    "npu": "huawei",
    "musa": "moore_threads",
    "mlu": "cambricon",
    "hygon": "hygon",
    "metax": "metax",
    "iluvatar": "iluvatar",
    "kunlunxin": "kunlunxin",
    "thead": "thead",
    "txda": "tsingmicro",
    "enflame": "enflame",
}

_RUNTIMES = {
    "cuda": "cuda",
    "npu": "cann",
    "musa": "musa",
    "mlu": "cnrt",
    "hygon": "rocm",
    "metax": "cuda-compatible",
    "iluvatar": "cuda-compatible",
    "kunlunxin": "cuda-compatible",
    "thead": "ppu-sdk",
    "txda": "txda",
    "enflame": "topsrt",
}

_COMPILERS = {
    "cuda": "triton",
    "npu": "triton-ascend",
    "musa": "triton-musa",
    "mlu": "triton-mlu",
    "hygon": "triton",
    "metax": "triton",
    "iluvatar": "triton",
    "kunlunxin": "triton",
    "thead": "triton",
    "txda": "triton-txda",
    "enflame": "flagtree",
}


def visibility_variables(backend: str) -> tuple[str, ...]:
    """Return device-visibility variables understood by one logical backend."""
    try:
        return _BACKEND_VISIBILITY_VARIABLES[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported backend for device binding: {backend!r}") from exc


def bind_logical_device(
    env: MutableMapping[str, str],
    variable: str,
    device: str,
) -> str:
    """Narrow one visibility variable from a logical device to one physical token."""
    logical_index = int(device.split(":", 1)[-1]) if ":" in device else 0
    configured = [item.strip() for item in env.get(variable, "").split(",") if item.strip()]
    if configured:
        if logical_index >= len(configured):
            raise ValueError(
                f"logical device {device!r} is outside {variable}={env[variable]!r}"
            )
        selected = configured[logical_index]
    else:
        selected = str(logical_index)
    env[variable] = selected
    return selected


def bind_backend_device(
    env: MutableMapping[str, str],
    backend: str,
    device: str,
) -> dict[str, str]:
    """Restrict a child environment to the assigned logical device.

    Some installations configure more than one equivalent visibility variable.
    Every configured variable is narrowed so their intersection continues to
    identify the same device.
    """
    candidates = visibility_variables(backend)
    configured = [name for name in candidates if env.get(name, "").strip()]
    selected_variables = configured or [candidates[0]]
    return {
        name: bind_logical_device(env, name, device)
        for name in selected_variables
    }


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def _module_version(name: str) -> str:
    """Read an importable module version without importing it in the server."""
    script = (
        "import importlib; "
        # Match worker import order. Importing Ascend Triton before Torch can
        # recurse through torch_npu into a partially initialized Triton module.
        + ("importlib.import_module('torch'); " if name == "triton" else "")
        + "import json; "
        f"module = importlib.import_module({name!r}); "
        "print('\\nKGS_MODULE_VERSION=' + json.dumps(getattr(module, '__version__', '')))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("KGS_MODULE_VERSION="):
            try:
                value = json.loads(line.removeprefix("KGS_MODULE_VERSION="))
            except json.JSONDecodeError:
                return ""
            return _clean_metadata_value(value) if isinstance(value, str) else ""
    return ""


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_names_for_module(name: str) -> tuple[str, ...]:
    """Return installed distributions that claim one importable module."""
    packages_distributions = getattr(
        importlib.metadata,
        "packages_distributions",
        None,
    )
    if callable(packages_distributions):
        try:
            names = packages_distributions().get(name, ())
        except (OSError, TypeError, ValueError):
            names = ()
        if names:
            return tuple(dict.fromkeys(str(item) for item in names if item))

    discovered: list[str] = []
    try:
        distributions = importlib.metadata.distributions()
        for distribution in distributions:
            try:
                top_level = distribution.read_text("top_level.txt") or ""
                owns_module = name in {
                    item.strip() for item in top_level.splitlines() if item.strip()
                }
                if not owns_module:
                    owns_module = any(
                        str(path).replace("\\", "/").split("/", 1)[0] == name
                        for path in (distribution.files or ())
                    )
                distribution_name = distribution.metadata.get("Name", "")
            except (KeyError, OSError, TypeError, ValueError):
                continue
            if (
                owns_module
                and distribution_name
                and distribution_name not in discovered
            ):
                discovered.append(distribution_name)
    except (OSError, TypeError, ValueError):
        return ()
    return tuple(discovered)


def _compiler_info(backend: str) -> tuple[str, str]:
    """Identify the installed distribution that provides the Triton compiler."""
    fallback = _COMPILERS.get(backend, "triton")
    owners = _distribution_names_for_module("triton")
    fallback_key = _canonical_distribution_name(fallback)
    selected = next(
        (
            owner
            for owner in owners
            if _canonical_distribution_name(owner) == fallback_key
        ),
        owners[0] if owners else fallback,
    )
    return _canonical_distribution_name(selected), _package_version(selected)


def _probe_environment(backend: str, devices: Sequence[str]) -> dict[str, object]:
    """Query devices and vendor metadata in one disposable child process."""
    script = """
import json
from dataclasses import asdict
from kernelgen_server.runtime.device import _make_device

backend = __import__("os").environ["KGS_PROBE_BACKEND"]
devices = json.loads(__import__("os").environ["KGS_PROBE_DEVICES"])
adapter = _make_device(backend)
items = []
for logical_device in devices:
    try:
        info = asdict(adapter.get_device_info(logical_device))
        items.append({"logical_device": logical_device, **info})
    except Exception as exc:
        items.append({"logical_device": logical_device, "error": f"{type(exc).__name__}: {exc}"})
try:
    metadata = adapter.status_metadata()
except Exception as exc:
    metadata = {"error": f"{type(exc).__name__}: {exc}"}
payload = {"devices": items, "metadata": metadata}
print("KGS_ENV_JSON=" + json.dumps(payload, separators=(",", ":")))
"""
    env = dict(os.environ)
    env["KGS_PROBE_BACKEND"] = backend
    env["KGS_PROBE_DEVICES"] = json.dumps(list(devices))
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            # Vendor Torch imports can be slow on a cold target. Keep this
            # aligned with the normal strong-probe startup budget.
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"devices": [], "metadata": {}, "error": "probe process failed"}
    if result.returncode != 0:
        return {"devices": [], "metadata": {}, "error": "probe process failed"}
    for line in reversed(result.stdout.splitlines()):
        if not line.startswith("KGS_ENV_JSON="):
            continue
        try:
            value = json.loads(line.removeprefix("KGS_ENV_JSON="))
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            return value
        break
    return {"devices": [], "metadata": {}, "error": "invalid probe output"}


def _clean_metadata_value(value: object) -> str:
    text = "" if value is None else str(value).strip()
    normalized = re.sub(r"\s+", " ", text.casefold()).strip(" :-")
    if normalized in {
        "",
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
    }:
        return ""
    if normalized.startswith(("n/a ", "not available ", "not found ")):
        return ""
    return text


_STATUS_OVERRIDES = {
    "target.architecture": "KGS_TARGET_ARCHITECTURE",
    "software.runtime_version": "KGS_RUNTIME_VERSION",
    "software.driver_version": "KGS_DRIVER_VERSION",
}

# Human-facing product labels only. These values must never replace a probed
# device identity or participate in target selection and execution policy.
DEVICE_DISPLAY_NAMES: dict[tuple[str, str], str] = {
    ("enflame", "ZIXIAOC200"): "S60",
}

_REQUIRED_STATUS_FIELDS = (
    "target.backend",
    "target.architecture",
    "target.device",
    "software.language",
    "software.language_version",
    "software.compiler",
    "software.compiler_version",
    "software.runtime",
    "software.runtime_version",
    "software.driver_version",
)


def environment_info(backend: str, devices: Sequence[str]) -> dict[str, object]:
    """Return best-effort target and software metadata for ``/status``."""
    language_version = _module_version("triton")
    compiler, compiler_version = _compiler_info(backend)
    compiler_version_source = (
        "python package metadata"
        if compiler_version
        else ""
    )
    probe = _probe_environment(backend, devices)
    probed_devices = probe.get("devices", [])
    target_devices = (
        [dict(item) for item in probed_devices if isinstance(item, dict)]
        if isinstance(probed_devices, list)
        else []
    )
    probed_metadata = probe.get("metadata", {})
    if not isinstance(probed_metadata, dict):
        probed_metadata = {}
    probed_sources = probed_metadata.get("sources", {})
    if not isinstance(probed_sources, dict):
        probed_sources = {}
    first = next(
        (item for item in target_devices if _clean_metadata_value(item.get("name"))),
        target_devices[0] if target_devices else {},
    )

    architecture = _clean_metadata_value(probed_metadata.get("architecture"))
    architecture_source = _clean_metadata_value(
        probed_sources.get("target.architecture")
    )
    if not architecture:
        architecture = _clean_metadata_value(first.get("compute_capability"))
        if architecture:
            architecture_source = "device runtime properties"

    target = {
        "backend": "ascend" if backend == "npu" else backend,
        "vendor": _VENDORS.get(backend, ""),
        "device": _clean_metadata_value(first.get("name")),
        "architecture": architecture,
        "devices": target_devices,
    }
    display_name = DEVICE_DISPLAY_NAMES.get((backend, target["device"]))
    if display_name:
        target["display_name"] = display_name
    software = {
        "language": "triton",
        "language_version": language_version,
        "compiler": compiler,
        "compiler_version": compiler_version,
        "runtime": _RUNTIMES.get(backend, ""),
        "runtime_version": _clean_metadata_value(
            probed_metadata.get("runtime_version")
        ),
        "driver_version": _clean_metadata_value(
            probed_metadata.get("driver_version")
        ),
    }
    sources = {
        "target.backend": "server configuration",
        "target.vendor": "server configuration",
        "software.language": "server configuration",
        "software.compiler": "python package metadata",
        "software.runtime": "server configuration",
    }
    if target["device"]:
        sources["target.device"] = "device probe"
    if target.get("display_name"):
        sources["target.display_name"] = "static display mapping"
    if target["architecture"] and architecture_source:
        sources["target.architecture"] = architecture_source
    if language_version:
        sources["software.language_version"] = "triton module metadata"
    if compiler_version:
        sources["software.compiler_version"] = compiler_version_source
    for path in ("software.runtime_version", "software.driver_version"):
        field = path.rsplit(".", 1)[-1]
        if software[field]:
            sources[path] = (
                _clean_metadata_value(probed_sources.get(path)) or "runtime probe"
            )

    for path, variable in _STATUS_OVERRIDES.items():
        override = _clean_metadata_value(os.environ.get(variable))
        if not override:
            continue
        section, field = path.split(".", 1)
        (target if section == "target" else software)[field] = override
        sources[path] = f"environment:{variable}"

    values = {"target": target, "software": software}
    missing = []
    for path in _REQUIRED_STATUS_FIELDS:
        section, field = path.split(".", 1)
        if not _clean_metadata_value(values[section].get(field)):
            missing.append(path)

    probe_errors = []
    if probe.get("error"):
        probe_errors.append(str(probe["error"]))
    if probed_metadata.get("error"):
        probe_errors.append(str(probed_metadata["error"]))
    probe_errors.extend(
        f"{item.get('logical_device', 'device')}: {item['error']}"
        for item in target_devices
        if item.get("error")
    )
    metadata = {
        "complete": not missing,
        "missing": missing,
        "sources": sources,
    }
    if probe_errors:
        metadata["probe_errors"] = probe_errors
    return {"target": target, "software": software, "metadata": metadata}
