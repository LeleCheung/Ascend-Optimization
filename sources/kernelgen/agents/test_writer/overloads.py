"""Overload schema query for a torch.ops.aten operator.

Used by the TestWriter agent to enumerate the overloads of a given operator and
cross-check them against PyTorch's native_functions.yaml.

Two independent sources are used:
  1. Runtime: ``torch.ops.aten.<op>`` overloadpacket's ``.overloads()`` + ``._schema``
  2. Static:  ``native_functions.yaml`` entries (grep by function name)

The runtime source is authoritative for "what can actually be called"; the YAML
source cross-validates and gives the full declaration text (including variant
defs) for prompt context.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

_NATIVE_FUNCTIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / ".claude" / "references" / "native_functions.yaml"
)


def query_aten_schemas(operator: str) -> List[str]:
    """Return the full schema strings of all overloads of ``operator``.

    Looks up ``torch.ops.aten.<operator>`` and returns e.g.
    ``["aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor", ...]``.
    Returns an empty list when torch is unavailable or the operator is unknown.
    """
    try:
        import torch
    except Exception:
        return []

    packet = getattr(torch.ops.aten, operator, None)
    if packet is None:
        return []

    overloads: List[str] = []
    try:
        names = packet.overloads()
    except Exception:
        names = ["default"]
    for name in names:
        try:
            fn = getattr(packet, name)
            overloads.append(f"{fn._schema}")
        except Exception:
            continue
    return overloads


def query_native_entries(operator: str) -> str:
    """Extract all native_functions.yaml entries matching ``operator``.

    Returns the raw YAML blocks (func + variants) joined by blank lines, or ""
    when the reference file is absent or no entry matches.
    """
    if not _NATIVE_FUNCTIONS_PATH.exists():
        return ""
    lines = _NATIVE_FUNCTIONS_PATH.read_text(encoding="utf-8").splitlines()
    entries: List[str] = []
    current: List[str] = []
    capturing = False
    for line in lines:
        if line.startswith("- func:"):
            if capturing and current:
                entries.append("\n".join(current))
            func_name = line.split("(")[0].replace("- func: ", "").split(".")[0]
            capturing = func_name == operator or func_name == f"_{operator}"
            current = [line] if capturing else []
        elif capturing:
            if line and not line.startswith("- func:"):
                current.append(line)
    if capturing and current:
        entries.append("\n".join(current))
    return "\n\n".join(entries)


def operator_exists(operator: str) -> bool:
    """True if the operator exists at runtime OR in native_functions.yaml."""
    if query_aten_schemas(operator):
        return True
    if query_native_entries(operator):
        return True
    return False


def format_overloads(operator: str) -> str:
    """Human-readable overload block for prompt injection.

    Returns something like::

        <overloads>
        aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
        aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
        </overloads>
    """
    schemas = query_aten_schemas(operator)
    if not schemas:
        return ""
    body = "\n".join(schemas)
    return f"<overloads>\n{body}\n</overloads>"
