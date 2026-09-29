"""Conservative detection of ambiguous view/layout usage in Triton submissions.

This is the complement of ``hack_detection.py``: it enumerates which
**ambiguous view/layout operations** (reshape, transpose, stride, copy,
broadcast, metadata queries) a submitted Triton implementation uses.

**IMPORTANT**: Tensor creation operations (zeros, ones, empty, rand) are
explicitly excluded from the graylist. These are not "ambiguous plumbing" but
clear allocation calls that a Triton implementation should handle via
``tl.zeros`` / pre-allocated outputs, not by calling torch APIs.

Detection is graylist-based: the ``graylists/*.yaml`` files enumerate the
legitimate view/layout callables for each Torch namespace (``torch``, ``torch.
nn.functional``, ``torch.linalg``, ``torch.fft``) plus ``torch.Tensor``
methods. These are computed as:

    graylist(namespace) = public_callables(namespace) - blacklist.forbidden(namespace) - tensor_creation_ops

See ``graylists/_generate.py`` and ``graylists/README.md`` for how the lists
are produced and maintained.

Two spellings are covered (same resolution as ``hack_detection.py``):

* Namespace form -- ``torch.clone(x)``, ``F.pad(x, ...)``,
  ``torch.linalg.diagonal(x)``.  Import aliases are resolved so ``import torch
  as tr; tr.transpose(...)`` and ``from torch import reshape as r; r(...)``
  both canonicalise to the ``torch.*`` namespace.
* Method form -- ``x.reshape(-1)``, ``x.contiguous()``.  The receiver type is
  unknown statically, so this matches the bare attribute name against the
  ``torch.Tensor`` graylist.  This is a best-effort signal (a same-named method
  on a non-Tensor object would also match), hence the "informational only"
  status.

Graylist hits are a **weak, informational signal**, not a fallback: a Triton
implementation is free to call ``torch.reshape(...)`` or ``x.contiguous()``.
Surfacing which view/layout operations a submission leans on is useful review
context, but should **never** be used to reject or penalise a submission.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..protocol.schema import Implementation, Language

_GRAYLIST_DIR = Path(__file__).with_name("graylists")


@dataclass(frozen=True)
class _Graylist:
    """Parsed contents of one ``graylists/*.yaml`` file."""

    namespace: str
    call_prefix: str
    gray: frozenset[str]


def _parse_graylist_yaml(path: Path) -> _Graylist:
    """Minimal parser for the fixed-shape graylist YAML.

    The files are machine-generated with a stable layout (``namespace:``,
    ``call_prefix:`` scalars followed by a ``gray:`` list of ``  - name``
    items), so a dependency-free line parser is used instead of pulling PyYAML
    into the core detection path.
    """
    namespace = ""
    call_prefix = ""
    gray: list[str] = []
    in_gray = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("namespace:"):
            namespace = line.split(":", 1)[1].strip().strip('"')
            in_gray = False
        elif line.startswith("call_prefix:"):
            call_prefix = line.split(":", 1)[1].strip().strip('"')
            in_gray = False
        elif line.startswith("gray:"):
            in_gray = True
        elif in_gray and line.lstrip().startswith("- "):
            gray.append(line.lstrip()[2:].strip())
    return _Graylist(namespace, call_prefix, frozenset(gray))


@lru_cache(maxsize=1)
def _load_graylists() -> dict[str, _Graylist]:
    """Load every ``graylists/*.yaml`` file, keyed by namespace.

    Cached for the process lifetime; the YAML files are static data.
    """
    lists: dict[str, _Graylist] = {}
    if _GRAYLIST_DIR.is_dir():
        for path in sorted(_GRAYLIST_DIR.glob("*.yaml")):
            gl = _parse_graylist_yaml(path)
            if gl.namespace:
                lists[gl.namespace] = gl
    return lists


@lru_cache(maxsize=1)
def _gray_full_names() -> frozenset[str]:
    """Canonical dotted names for namespace-form calls.

    Combines every ``call_prefix + name`` for the prefixed namespaces
    (``torch.*``, ``torch.linalg.*``, ``torch.fft.*``).  ``nn.functional`` is
    included under its canonical ``torch.nn.functional.`` prefix so aliases
    that resolve there are caught, regardless of the local ``F.`` spelling.
    """
    names: set[str] = set()
    for gl in _load_graylists().values():
        if gl.namespace == "torch.Tensor":
            continue  # method form, handled separately
        prefix = "torch.nn.functional." if gl.namespace == "torch.nn.functional" else gl.call_prefix
        for name in gl.gray:
            names.add(prefix + name)
    return frozenset(names)


@lru_cache(maxsize=1)
def _gray_tensor_methods() -> frozenset[str]:
    """Bare method names from the ``torch.Tensor`` graylist."""
    gl = _load_graylists().get("torch.Tensor")
    return gl.gray if gl else frozenset()


@dataclass(frozen=True)
class GraylistDetection:
    """Result of graylist detection for a submitted Triton implementation.

    ``gray_calls`` and ``gray_methods`` are tuples of (file_path, line_number,
    callable_name) recording which legitimate plumbing the submission uses.
    """

    gray_calls: tuple[tuple[str, int, str], ...]
    gray_methods: tuple[tuple[str, int, str], ...]

    def has_hits(self) -> bool:
        """True if any graylist plumbing was detected."""
        return bool(self.gray_calls or self.gray_methods)

    def summary(self) -> str:
        """Human-readable summary of graylist hits, or empty string if none."""
        reasons = [
            f"graylist Torch API: {path}:{line} {call_name}"
            for path, line, call_name in sorted(self.gray_calls)
        ]
        reasons += [
            f"graylist Tensor method: {path}:{line} .{method}"
            for path, line, method in sorted(self.gray_methods)
        ]
        return "; ".join(reasons)


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _module_aliases(tree: ast.AST, module: str) -> dict[str, str]:
    """Collect import aliases that resolve onto the given module.

    Examples:
        import torch as tr           → {"tr": "torch"}
        import torch                 → {"torch": "torch"}
        import torch.linalg          → {"torch": "torch"}  (binds "torch", not "torch.linalg")
        from torch import zeros as z → {"z": "torch.zeros"}
        from torch import matmul     → {"matmul": "torch.matmul"}
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == module or item.name.startswith(f"{module}."):
                    if item.asname:
                        # import torch as tr → {"tr": "torch"}
                        # import torch.linalg as tl → {"tl": "torch.linalg"}
                        aliases[item.asname] = item.name
                    else:
                        # import torch / import torch.linalg both bind "torch"
                        local_name = item.name.split(".", 1)[0]
                        # Only record if the root matches the target module
                        # (avoid recording {"torch": "torch.linalg"} when module="torch.linalg")
                        if local_name == module:
                            aliases[local_name] = module
        elif isinstance(node, ast.ImportFrom) and (
            node.module == module or (node.module or "").startswith(f"{module}.")
        ):
            for item in node.names:
                if item.name == "*":
                    continue
                local_name = item.asname or item.name
                aliases[local_name] = f"{node.module}.{item.name}"
    return aliases


def _canonical_name(name: str | None, aliases: dict[str, str]) -> str | None:
    if name is None:
        return None
    root, separator, remainder = name.partition(".")
    canonical_root = aliases.get(root)
    if canonical_root is None:
        return name
    return canonical_root + (separator + remainder if separator else "")


def _imported_names(tree: ast.AST) -> frozenset[str]:
    """Local names bound by ``import`` / ``from ... import`` statements.

    These are module namespaces (``math``, ``tl`` from ``import
    triton.language as tl``, ``libdevice`` from ``from triton.language.extra
    import libdevice``), never Tensor instances.  Used to distinguish a
    namespace call such as ``tl.math.tanh(x)`` from a genuine Tensor method
    call such as ``x.tanh()``.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                names.add(item.asname or item.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                if item.name != "*":
                    names.add(item.asname or item.name)
    return frozenset(names)


def _receiver_is_namespace(receiver: ast.AST, imported: frozenset[str]) -> bool:
    """True if a method receiver is a module namespace rather than a value.

    Walks the attribute chain of the receiver to its root name.  A rooted
    import (``tl`` in ``tl.math.tanh``, ``libdevice`` in ``libdevice.tanh``,
    ``math`` in ``math.log``) is a namespace-form call and must not be flagged
    as a Tensor method.  A bare local variable (``x`` in ``x.tanh()``) or any
    other expression (``foo().tanh()``) is a runtime value and stays subject
    to the Tensor-method check.
    """
    node = receiver
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in imported


def detect_graylist_usage(implementation: Implementation) -> GraylistDetection:
    """Label legitimate plumbing usage in a submitted Triton implementation."""
    if implementation.language != Language.TRITON:
        return GraylistDetection(gray_calls=(), gray_methods=())

    gray_full_names = _gray_full_names()
    gray_tensor_methods = _gray_tensor_methods()

    gray_calls: set[tuple[str, int, str]] = set()
    gray_methods: set[tuple[str, int, str]] = set()

    for source in implementation.sources:
        if not source.path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source.content, filename=source.path)
        except SyntaxError:
            # The regular implementation loader remains authoritative for
            # syntax and import errors.
            continue

        # Resolve aliases for every namespace we graylist so that both
        # ``import torch as tr`` and ``from torch import zeros as z`` spellings
        # canonicalise onto the dotted names in ``gray_full_names``.
        aliases: dict[str, str] = {}
        for module in ("torch", "torch.nn.functional", "torch.linalg", "torch.fft"):
            aliases.update(_module_aliases(tree, module))

        # Local names bound to imported modules.  A method receiver rooted at
        # one of these is a namespace call (``tl.math.tanh``, ``libdevice.exp``,
        # ``math.log``), not a Tensor method, and must be exempt below.
        imported = _imported_names(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = _canonical_name(_dotted_name(node.func), aliases)
                if call_name is not None and call_name in gray_full_names:
                    gray_calls.add(
                        (source.path, node.lineno, call_name)  # type: ignore[arg-type]
                    )

                # Method form: ``x.reshape(...)``.  The receiver is a runtime
                # value (a variable, or the result of a chained call), so this
                # matches the bare attribute name against the Tensor graylist.
                # Also skip receivers rooted at an imported module
                # (``tl.math.tanh``, ``libdevice.exp``, ``math.log``): those
                # are Triton/stdlib namespace calls, not Tensor methods.
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in gray_tensor_methods
                    and not _receiver_is_namespace(node.func.value, imported)
                ):
                    gray_methods.add(
                        (source.path, node.lineno, node.func.attr)
                    )

    return GraylistDetection(
        gray_calls=tuple(sorted(gray_calls)),
        gray_methods=tuple(sorted(gray_methods)),
    )
