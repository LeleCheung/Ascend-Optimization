"""Deterministic FlagGems checkout and pytest-asset discovery."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FlagGemsAssets:
    root: Path
    correctness: tuple[Path, ...]
    performance: tuple[Path, ...]
    native_operator: str
    required_revision: str | None
    revision: str
    pytest_marker: str | None = None


_PYTEST_MARKER_ALIASES = {
    "_flash_attention_forward": ("underscore_flash_attention_forward",),
}

_REPRODUCIBILITY_PATHS = (
    "*.py",
    "*.pyi",
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
)


def flag_gems_root() -> Path:
    configured = os.environ.get("KGS_FLAGGEMS_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path.cwd(),
            Path("/data/FlagGems-master"),
            Path("/data/akg_kernel_bench_lite/FlagGems-master"),
            Path("/workspace/FlagGems-master"),
            Path("/workspace/FlagGems"),
        ]
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (
            (root / "benchmark/base.py").is_file()
            and (root / "benchmark/conftest.py").is_file()
            and (root / "src/flag_gems/__init__.py").is_file()
        ):
            return root
    raise RuntimeError(
        "FlagGems source tree is unavailable; set KGS_FLAGGEMS_ROOT to the pinned checkout"
    )


def _operator_aliases(operator: str) -> tuple[str, ...]:
    """Return deterministic pytest-marker aliases for public catalog ids.

    Most KernelGen names are already the FlagGems marker/op_name.  A small
    historical set used an implementation spelling such as ``ilshift__`` while
    the native suite uses ``ilshift``.  Stripping only edge underscores keeps
    this compatibility deterministic without introducing a per-operator
    registry.
    """

    aliases = [operator, *_PYTEST_MARKER_ALIASES.get(operator, ())]
    stripped = operator.strip("_")
    if stripped and stripped not in aliases:
        aliases.append(stripped)
    return tuple(aliases)


def _pytest_mark_names(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    }


@lru_cache(maxsize=None)
def _suite_marker_index(root: Path, suite: str) -> dict[str, tuple[Path, ...]]:
    index: dict[str, list[Path]] = {}
    suite_root = root / suite
    if not suite_root.is_dir():
        return {}
    for path in sorted(suite_root.glob("test_*.py")):
        for marker in _pytest_mark_names(path):
            index.setdefault(marker, []).append(path)
    return {marker: tuple(paths) for marker, paths in index.items()}


def _discover_suite(
    root: Path, suite: str, operator: str
) -> tuple[tuple[Path, ...], str]:
    aliases = _operator_aliases(operator)
    index = _suite_marker_index(root, suite)
    for alias in aliases:
        marked = index.get(alias, ())
        if marked:
            return marked, alias

    for alias in aliases:
        exact = root / suite / f"test_{alias}.py"
        if exact.is_file():
            return (exact,), alias
    return (), operator


def discover_assets(
    operator: str, required_revision: str | None = None
) -> FlagGemsAssets:
    """Read runtime assets; only frozen native oracles supply a revision pin."""
    root = flag_gems_root()
    revision = _checkout_revision(root)
    _validate_checkout(root, required_revision, revision)
    correctness, correctness_operator = _discover_suite(root, "tests", operator)
    performance, performance_operator = _discover_suite(
        root, "benchmark", operator
    )
    if not correctness or not performance:
        missing = []
        if not correctness:
            missing.append(f"tests marker/file for {operator!r}")
        if not performance:
            missing.append(f"benchmark marker/file for {operator!r}")
        raise RuntimeError(
            "FlagGems adapter could not discover pytest assets: "
            + ", ".join(missing)
        )
    if correctness_operator != performance_operator:
        raise RuntimeError(
            "FlagGems correctness/performance marker mismatch: "
            f"{correctness_operator!r} != {performance_operator!r}"
        )
    return FlagGemsAssets(
        root,
        correctness,
        performance,
        operator,
        required_revision,
        revision,
        pytest_marker=correctness_operator,
    )


def _checkout_revision(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "FlagGems source tree must be a Git checkout with a committed HEAD"
        )
    return process.stdout.strip()


def _validate_checkout(root: Path, required_revision: str | None, revision: str) -> None:
    if required_revision is not None:
        pinned = subprocess.run(
            ["git", "rev-parse", "--verify", f"{required_revision}^{{commit}}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if pinned.returncode != 0:
            raise RuntimeError(
                "FlagGems checkout does not contain the required fixed revision: "
                f"{required_revision}"
            )
        pinned_revision = pinned.stdout.strip()
        if revision != pinned_revision:
            raise RuntimeError(
                "FlagGems checkout HEAD does not match the required fixed revision: "
                f"required={pinned_revision}, checkout={revision}"
            )

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise RuntimeError(
            "FlagGems clean-worktree validation failed: "
            + (status.stderr.strip() or "git status failed")
        )
    dirty = status.stdout.splitlines()
    if not dirty:
        return

    reproducibility_status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *_REPRODUCIBILITY_PATHS,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if reproducibility_status.returncode != 0:
        raise RuntimeError(
            "FlagGems source-worktree validation failed: "
            + (reproducibility_status.stderr.strip() or "git status failed")
        )
    source_dirty = reproducibility_status.stdout.splitlines()
    if source_dirty:
        preview = ", ".join(line.strip() for line in source_dirty[:5])
        if len(source_dirty) > 5:
            preview += f", ... ({len(source_dirty)} entries)"
        raise RuntimeError(
            "FlagGems checkout has uncommitted changes affecting Python source "
            "or pytest configuration; restore the selected revision "
            f"{revision} before starting KGS: {preview}"
        )

    preview = ", ".join(line.strip() for line in dirty[:5])
    if len(dirty) > 5:
        preview += f", ... ({len(dirty)} entries)"
    warnings.warn(
        "FlagGems checkout has ignored non-source changes: " + preview,
        RuntimeWarning,
        stacklevel=2,
    )


def benchmark_fingerprint(
    assets: FlagGemsAssets,
    *,
    operator: str,
    benchmark_level: str,
    native_case_report: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(assets.revision.encode())
    digest.update(operator.encode())
    digest.update(benchmark_level.encode())
    paths = (
        *assets.correctness,
        *assets.performance,
        assets.root / "benchmark/base.py",
        assets.root / "benchmark/conftest.py",
    )
    for path in dict.fromkeys(paths):
        digest.update(path.relative_to(assets.root).as_posix().encode())
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(
            native_case_report,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return "sha256:" + digest.hexdigest()


__all__ = [
    "FlagGemsAssets",
    "benchmark_fingerprint",
    "discover_assets",
    "flag_gems_root",
]
