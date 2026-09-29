"""Deterministic path matching for SourcePackage ingestion rules."""

from __future__ import annotations

from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import PurePosixPath


def source_path_matches(path: str, pattern: str) -> bool:
    """Match POSIX paths with segment-aware ``*`` and recursive ``**``.

    A pattern without ``/`` retains the existing basename behavior, so the
    baseline ``*`` rule classifies every manifest entry.
    """

    normalized_path = PurePosixPath(path)
    normalized_pattern = PurePosixPath(pattern)
    if "/" not in pattern:
        return fnmatchcase(normalized_path.name, pattern)

    path_parts = normalized_path.parts
    pattern_parts = normalized_pattern.parts

    @lru_cache(maxsize=None)
    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        part = pattern_parts[pattern_index]
        if part == "**":
            return match(path_index, pattern_index + 1) or (
                path_index < len(path_parts)
                and match(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], part)
            and match(path_index + 1, pattern_index + 1)
        )

    return match(0, 0)
