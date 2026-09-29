"""Small PyTree helpers independent of a particular PyTorch release."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple


Path = Tuple[Any, ...]


@dataclass(frozen=True)
class Leaf:
    path: Path
    value: Any


def _is_dataclass_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def leaves(tree: Any, path: Path = ()) -> Iterator[Leaf]:
    # Dataclasses (e.g. the LoRA ``LoRABatchInfo`` input) hold tensors and
    # scalars; descend into their fields so mutation checks and comparisons
    # see the actual values instead of comparing the object identity.
    if _is_dataclass_instance(tree):
        for field in dataclasses.fields(tree):
            yield from leaves(getattr(tree, field.name), path + (field.name,))
        return
    if isinstance(tree, tuple):
        for index, value in enumerate(tree):
            yield from leaves(value, path + (index,))
        return
    if isinstance(tree, list):
        for index, value in enumerate(tree):
            yield from leaves(value, path + (index,))
        return
    if isinstance(tree, Mapping):
        for key in sorted(tree):
            yield from leaves(tree[key], path + (key,))
        return
    yield Leaf(path, tree)


def structure(tree: Any) -> Any:
    if _is_dataclass_instance(tree):
        return (
            "dataclass",
            tuple(
                (field.name, structure(getattr(tree, field.name)))
                for field in dataclasses.fields(tree)
            ),
        )
    if isinstance(tree, tuple):
        return ("tuple", tuple(structure(value) for value in tree))
    if isinstance(tree, list):
        return ("list", tuple(structure(value) for value in tree))
    if isinstance(tree, Mapping):
        return ("dict", tuple((key, structure(tree[key])) for key in sorted(tree)))
    return "leaf"


def clone(tree: Any, *, device: str | None = None) -> Any:
    """Clone a tree while preserving repeated-object aliases and lazy neg views."""

    memo: Dict[int, Any] = {}

    def visit(value: Any) -> Any:
        value_id = id(value)
        if value_id in memo:
            return memo[value_id]

        try:
            import torch
        except ImportError:  # pragma: no cover - runtime package supplies torch
            torch = None

        if torch is not None and isinstance(value, torch.Tensor):
            was_neg = bool(value.is_neg())
            target = value.detach()
            if device is not None:
                target = target.to(device)
            cloned = target.clone(memory_format=torch.preserve_format)
            if was_neg:
                cloned = torch._neg_view(cloned.neg())
            cloned.requires_grad_(value.requires_grad)
            memo[value_id] = cloned
            return cloned
        if isinstance(value, list):
            result: List[Any] = []
            memo[value_id] = result
            result.extend(visit(item) for item in value)
            return result
        if isinstance(value, tuple):
            result = tuple(visit(item) for item in value)
            memo[value_id] = result
            return result
        if isinstance(value, dict):
            result: Dict[Any, Any] = {}
            memo[value_id] = result
            result.update((copy.deepcopy(key), visit(item)) for key, item in value.items())
            return result
        return copy.deepcopy(value, memo)

    return visit(tree)


def to_cpu(tree: Any) -> Any:
    return clone(tree, device="cpu")
