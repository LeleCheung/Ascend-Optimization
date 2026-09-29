"""Read-only source views for vendor report parsers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceView:
    path: str
    content: str


@dataclass(frozen=True)
class SourceImplementationView:
    sources: tuple[SourceView, ...]


@dataclass(frozen=True)
class ReportSourceTarget:
    """Minimal source-only facade consumed by legacy vendor report parsers."""

    implementation: SourceImplementationView


def source_views(roots: Iterable[Path]) -> list[SourceView]:
    result: list[SourceView] = []
    seen: set[str] = set()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in seen:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            seen.add(relative)
            result.append(SourceView(relative, content))
    return result


def report_source_target(roots: Iterable[Path]) -> ReportSourceTarget:
    """Compatibility facade for parsers that only inspect implementation.sources."""

    return ReportSourceTarget(
        implementation=SourceImplementationView(tuple(source_views(roots)))
    )


__all__ = [
    "ReportSourceTarget",
    "SourceImplementationView",
    "SourceView",
    "report_source_target",
    "source_views",
]
