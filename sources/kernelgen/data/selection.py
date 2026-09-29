"""Authoritative best-score selection."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, TypeVar

T = TypeVar("T")


def pick_best(
    candidates: List[T],
    key: Callable[[T], Any],
    default: Optional[T] = None,
) -> Optional[T]:
    """AUTHORITATIVE score winner: the candidate maximizing ``key(c)``.

    Ties resolve to the earliest-seen candidate (Python ``max`` semantics).
    Empty input returns ``default`` — the caller decides whether empty is an
    error. Red-line: Python calls this unconditionally; its output is the fact.
    """
    if not candidates:
        return default
    return max(candidates, key=key)
