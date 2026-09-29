"""Conservative detection of obvious framework fallbacks.

This is intentionally a high-confidence signal, not a security sandbox.  It
only labels submitted Triton implementations and does not block execution.

Detection is blacklist-based: the ``blacklists/*.yaml`` files enumerate the
forbidden compute callables for each Torch namespace (``torch``, ``torch.nn.
functional``, ``torch.linalg``, ``torch.fft``) plus ``torch.Tensor`` methods.
Legitimate plumbing (allocation, dtype/device movement, shape/layout,
autograd flags) is deliberately excluded from those lists.  See
``blacklists/_generate.py`` and ``blacklists/README.md`` for how the lists are
produced and maintained.

Two spellings are covered:

* Namespace form -- ``torch.matmul(a, b)``, ``F.relu(x)``,
  ``torch.linalg.svd(x)``.  Import aliases are resolved so ``import torch as
  tr; tr.matmul(...)`` and ``from torch import matmul as mm; mm(...)`` also
  canonicalise to ``torch.matmul``.
* Method form -- ``x.softmax(-1)``.  The receiver type is unknown statically,
  so this matches the bare attribute name against the ``torch.Tensor``
  blacklist.  This is a best-effort signal (a same-named method on a
  non-Tensor object would also match), which is why detection only labels and
  never blocks.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..protocol.schema import Implementation, Language
from .metadata_policy import permits_no_jit

logger = logging.getLogger(__name__)

_BLACKLIST_DIR = Path(__file__).with_name("blacklists")
_GRAYLIST_LOG_DIR = Path(__file__).with_name("graylists") / "log"

# Namespace prefixes that are forbidden wholesale.  These are dynamic dispatch
# surfaces that expose thousands of low-level ops and cannot be usefully
# enumerated (the set grows every Torch release), so any call underneath them
# is treated as a Torch fallback.  ``torch.ops.aten.matmul(...)`` /
# ``torch.ops.aten.add.Tensor(...)`` are the canonical escape hatch around the
# per-name ``torch.*`` blacklist.
_FORBIDDEN_PREFIXES = (
    "torch.ops.",
    "torch._C.",
    "torch.overrides.",
)


@dataclass(frozen=True)
class _Blacklist:
    """Parsed contents of one ``blacklists/*.yaml`` file."""

    namespace: str
    call_prefix: str
    forbidden: frozenset[str]


def _parse_blacklist_yaml(path: Path) -> _Blacklist:
    """Minimal parser for the fixed-shape blacklist YAML.

    The files are machine-generated with a stable layout (``namespace:``,
    ``call_prefix:`` scalars followed by a ``forbidden:`` list of ``  - name``
    items), so a dependency-free line parser is used instead of pulling PyYAML
    into the core detection path.
    """
    namespace = ""
    call_prefix = ""
    forbidden: list[str] = []
    in_forbidden = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("namespace:"):
            namespace = line.split(":", 1)[1].strip().strip('"')
            in_forbidden = False
        elif line.startswith("call_prefix:"):
            call_prefix = line.split(":", 1)[1].strip().strip('"')
            in_forbidden = False
        elif line.startswith("forbidden:"):
            in_forbidden = True
        elif in_forbidden and line.lstrip().startswith("- "):
            forbidden.append(line.lstrip()[2:].strip())
    return _Blacklist(namespace, call_prefix, frozenset(forbidden))


@lru_cache(maxsize=1)
def _load_blacklists() -> dict[str, _Blacklist]:
    """Load every ``blacklists/*.yaml`` file, keyed by namespace.

    Cached for the process lifetime; the YAML files are static data.
    """
    lists: dict[str, _Blacklist] = {}
    if _BLACKLIST_DIR.is_dir():
        for path in sorted(_BLACKLIST_DIR.glob("*.yaml")):
            bl = _parse_blacklist_yaml(path)
            if bl.namespace:
                lists[bl.namespace] = bl
    return lists


@lru_cache(maxsize=1)
def _forbidden_full_names() -> frozenset[str]:
    """Canonical dotted names for namespace-form calls.

    Combines every ``call_prefix + name`` for the prefixed namespaces
    (``torch.*``, ``torch.linalg.*``, ``torch.fft.*``).  ``nn.functional`` is
    included under its canonical ``torch.nn.functional.`` prefix so aliases
    that resolve there are caught, regardless of the local ``F.`` spelling.
    """
    names: set[str] = set()
    for bl in _load_blacklists().values():
        if bl.namespace == "torch.Tensor":
            continue  # method form, handled separately
        prefix = "torch.nn.functional." if bl.namespace == "torch.nn.functional" else bl.call_prefix
        for name in bl.forbidden:
            names.add(prefix + name)
    return frozenset(names)


@lru_cache(maxsize=1)
def _forbidden_tensor_methods() -> frozenset[str]:
    """Bare method names from the ``torch.Tensor`` blacklist."""
    bl = _load_blacklists().get("torch.Tensor")
    return bl.forbidden if bl else frozenset()


@dataclass(frozen=True)
class HackDetection:
    is_hack: bool
    hack_reason: str
    rejection_reasons: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()


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
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == module or item.name.startswith(f"{module}."):
                    local_name = item.asname or item.name.split(".", 1)[0]
                    aliases[local_name] = item.name if item.asname else module
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


def _decorator_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Call):
        node = node.func
    return _canonical_name(_dotted_name(node), aliases)


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


def detect_obvious_hack(
    implementation: Implementation, *, operator_name: str | None = None
) -> HackDetection:
    """Label obvious Torch fallbacks in a submitted Triton implementation."""
    if implementation.language != Language.TRITON:
        return HackDetection(False, "")

    forbidden_full_names = _forbidden_full_names()
    forbidden_tensor_methods = _forbidden_tensor_methods()

    forbidden_calls: set[tuple[str, int, str]] = set()
    forbidden_methods: set[tuple[str, int, str]] = set()
    forbidden_getattr: set[tuple[str, int, str]] = set()
    triton_kernels: set[str] = set()
    launch_targets: list[str] = []

    for source in implementation.sources:
        if not source.path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source.content, filename=source.path)
        except SyntaxError:
            # The regular implementation loader remains authoritative for
            # syntax and import errors.
            continue

        # Resolve aliases for every namespace we blacklist so that both
        # ``import torch as tr`` and ``from torch.linalg import svd`` spellings
        # canonicalise onto the dotted names in ``forbidden_full_names``.
        aliases: dict[str, str] = {}
        for module in ("torch", "torch.nn.functional", "torch.linalg", "torch.fft"):
            aliases.update(_module_aliases(tree, module))
        triton_aliases = _module_aliases(tree, "triton")

        # Local names bound to imported modules.  A method receiver rooted at
        # one of these is a namespace call (``tl.math.tanh``, ``libdevice.exp``,
        # ``math.log``), not a Tensor method, and must be exempt below.
        imported = _imported_names(tree)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(
                    _decorator_name(decorator, triton_aliases) == "triton.jit"
                    for decorator in node.decorator_list
                ):
                    triton_kernels.add(node.name)
            elif isinstance(node, ast.Call):
                call_name = _canonical_name(_dotted_name(node.func), aliases)
                already_flagged = call_name is not None and (
                    call_name in forbidden_full_names
                    or call_name.startswith(_FORBIDDEN_PREFIXES)
                )
                if already_flagged:
                    forbidden_calls.add(
                        (source.path, node.lineno, call_name)  # type: ignore[arg-type]
                    )

                # getattr(torch, "matmul") obfuscation detection
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    base = _canonical_name(_dotted_name(node.args[0]), aliases)
                    if base is not None:
                        attr_name = node.args[1].value
                        full = f"{base}.{attr_name}"
                        if full in forbidden_full_names or full.startswith(_FORBIDDEN_PREFIXES):
                            forbidden_getattr.add((source.path, node.lineno, full))

                # Method form: ``x.softmax(...)``.  The receiver is a runtime
                # value (a variable, or the result of a chained call), so this
                # matches the bare attribute name against the Tensor blacklist.
                # Guarded by ``not already_flagged`` so namespace forms like
                # ``torch.softmax(x)`` -- whose func also ends in a blacklisted
                # ``.softmax`` attribute -- are not double-counted.
                # Also skip receivers rooted at an imported module
                # (``tl.math.tanh``, ``libdevice.exp``, ``math.log``): those
                # are Triton/stdlib namespace calls, not Tensor methods.
                if (
                    isinstance(node.func, ast.Attribute)
                    and not already_flagged
                    and node.func.attr in forbidden_tensor_methods
                    and not _receiver_is_namespace(node.func.value, imported)
                ):
                    forbidden_methods.add(
                        (source.path, node.lineno, node.func.attr)
                    )

                if isinstance(node.func, ast.Subscript):
                    launch_name = _dotted_name(node.func.value)
                    if launch_name is not None:
                        launch_targets.append(launch_name.rsplit(".", 1)[-1])

    reasons = [
        f"forbidden Torch API: {path}:{line} {call_name}"
        for path, line, call_name in sorted(forbidden_calls)
    ]
    method_reasons = [
        f"forbidden Tensor method: {path}:{line} .{method}"
        for path, line, method in sorted(forbidden_methods)
    ]
    reasons += [
        f"getattr obfuscation: {path}:{line} {call_name}"
        for path, line, call_name in sorted(forbidden_getattr)
    ]
    metadata_only = permits_no_jit(implementation, operator_name)
    if not triton_kernels and not metadata_only:
        reasons.append("no @triton.jit kernel found")
    if not triton_kernels.intersection(launch_targets) and not metadata_only:
        reasons.append("no Triton kernel launch found")
    all_reasons = reasons + method_reasons
    return HackDetection(
        bool(all_reasons), "; ".join(all_reasons), tuple(reasons), tuple(method_reasons)
    )


def log_graylist_hits(
    implementation: Implementation,
    *,
    vendor: str | None = None,
    persist_dir: Path | str | None = "default",
) -> None:
    """Detect and log view/layout operations found in a Triton implementation.

    This is a **soft-signal companion** to ``detect_obvious_hack``.  It never
    raises, never sets any hack flag, and never blocks execution.  Its sole job
    is to emit a WARNING-level log entry listing every graylist hit so that
    operators can be reviewed manually later.

    Graylist entries are view/layout/copy/metadata ops (e.g. ``torch.reshape``,
    ``x.contiguous()``) that fall in an ambiguous zone: they are *not* forbidden
    compute operators, but they are also *not* pure Triton.  Whether a given
    submission should use them is left to human review.

    Args:
        implementation: The submitted Triton implementation to check.
        vendor: Chip vendor name (e.g. "haiguang", "muxi"). Optional, only used
            for persistent recording.
        persist_dir: Directory to write graylist hits to disk. Defaults to
            ``graylists/log/`` (relative to this module). Pass ``None`` to
            disable persistence and only log. Pass an explicit path to override
            the default location.

    Log format (one WARNING per submission with hits, DEBUG if clean):
        GRAYLIST [<name>] 3 hit(s): graylist Torch API: kernel.py:9 torch.clone; ...
        GRAYLIST [<name>] clean (no view/layout ops detected)
    """
    # Lazy import to keep the circular-import surface minimal and avoid
    # pulling graylist YAML into memory when graylist checking is not needed.
    from .graylist_detection import detect_graylist_usage  # noqa: PLC0415

    gray = detect_graylist_usage(implementation)
    tag = f"GRAYLIST [{implementation.name}]"
    if gray.has_hits():
        n = len(gray.gray_calls) + len(gray.gray_methods)
        logger.warning("%s %d hit(s): %s", tag, n, gray.summary())

        # Persist to disk if requested
        # persist_dir="default" → use _GRAYLIST_LOG_DIR
        # persist_dir=None → no persistence
        # persist_dir="/path" → explicit path
        effective_dir = _GRAYLIST_LOG_DIR if persist_dir == "default" else persist_dir
        if effective_dir is not None:
            _persist_graylist_hit(implementation, vendor, gray, effective_dir)
    else:
        logger.debug("%s clean (no view/layout ops detected)", tag)


def _persist_graylist_hit(
    implementation: Implementation,
    vendor: str | None,
    gray: "GraylistDetection",
    persist_dir: Path | str,
) -> None:
    """Write graylist detection result to a JSON file for manual review.

    Creates ``<vendor>_<operator>.json`` under ``persist_dir`` with the
    structure:
        {
          "operator": "<name>",
          "vendor": "<vendor>",
          "language": "triton",
          "hit_count": N,
          "gray_calls": [...],
          "gray_methods": [...]
        }
    """
    import json  # noqa: PLC0415

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    # Extract operator name from implementation.name
    # (implementation.name may be "vendor/op" or just "op")
    op_name = implementation.name.split("/")[-1] if "/" in implementation.name else implementation.name
    vendor_prefix = f"{vendor}_" if vendor else ""
    filename = f"{vendor_prefix}{op_name}.json"
    target = persist_path / filename

    record = {
        "operator": op_name,
        "vendor": vendor or "unknown",
        "language": implementation.language,
        "hit_count": len(gray.gray_calls) + len(gray.gray_methods),
        "gray_calls": [
            {"file": path, "line": line, "call": name}
            for path, line, name in gray.gray_calls
        ],
        "gray_methods": [
            {"file": path, "line": line, "method": method}
            for path, line, method in gray.gray_methods
        ],
    }

    target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("GRAYLIST [%s] persisted → %s", implementation.name, target)
