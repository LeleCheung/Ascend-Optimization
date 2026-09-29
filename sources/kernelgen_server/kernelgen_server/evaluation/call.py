"""Normalize executable operator calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .pytree import clone


@dataclass
class Call:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    def cloned(self, *, device: str | None = None) -> "Call":
        args, kwargs = clone((self.args, self.kwargs), device=device)
        return Call(tuple(args), dict(kwargs))

    def snapshot(self) -> Dict[str, Any]:
        return {"args": self.args, "kwargs": self.kwargs}


def normalize_call(value: Any) -> Call:
    if isinstance(value, dict) and set(value).issubset({"args", "kwargs"}):
        args = value.get("args", ())
        kwargs = value.get("kwargs", {})
        if not isinstance(args, (list, tuple)):
            raise TypeError("make_inputs result['args'] must be a list or tuple")
        if not isinstance(kwargs, dict):
            raise TypeError("make_inputs result['kwargs'] must be a mapping")
        return Call(tuple(args), dict(kwargs))
    if isinstance(value, tuple):
        return Call(value, {})
    if isinstance(value, list):
        return Call(tuple(value), {})
    return Call((value,), {})
