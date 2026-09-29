"""Deterministic FlagGems source discovery for the extraction agent.

This module is deliberately independent of the catalog schema and prompt.  It is
the repository adapter: given an operator name, it resolves the implementation,
pytest, benchmark, catalog metadata, and directly referenced helper source that
the semantic extractor is allowed to use.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .source_profile import checkout_profile


class OperatorNotImplementedError(ValueError):
    """Raised when an operator has no implementation or correctness source."""


class OperatorSourceNotFoundError(OperatorNotImplementedError):
    """Raised when no pytest exists to define correctness semantics."""


def clear_source_inventory_cache() -> None:
    """Start a new committed-source export after a checkout may have changed."""
    for function in (_operator_catalog, _exported_operator_symbols, _exported_operator_bindings,
                     _registered_operator_bindings, _exported_operator_modules, _suite_marker_files):
        function.cache_clear()


@dataclass(frozen=True)
class PublicPythonSignature:
    """A public export and a body-free copy of its source Python signature."""

    public_name: str
    implementation_name: str
    source: str


@dataclass(frozen=True)
class FlagGemsSourceInventory:
    """Resolved, source-backed inputs to one semantic extraction."""

    operator: str
    repo: Path
    labels: tuple[str, ...]
    benchmark_shapes: str | None
    operator_symbols: tuple[str, ...]
    implementation_file: Path
    public_signatures: tuple[PublicPythonSignature, ...]
    test_files: tuple[Path, ...]
    benchmark_files: tuple[Path, ...]
    test_file: Path
    benchmark_file: Path
    implemented: bool
    helper_context: str

    @property
    def implementation_status(self) -> str:
        if self.implemented:
            return "exported implementation"
        return "pytest baseline only; extract for a new FlagGems implementation"


def collect_source_inventory(
    flaggems_repo: str,
    operator: str,
) -> FlagGemsSourceInventory:
    """Resolve all authoritative source artifacts before invoking the LLM."""

    repo = Path(flaggems_repo)
    labels = _find_operator_labels(flaggems_repo, operator)
    operator_symbols = _operator_symbols(flaggems_repo, operator)
    implementation_file = _find_implementation_file(
        flaggems_repo,
        operator,
        operator_symbols,
    )
    test_files = _find_suite_files(
        flaggems_repo,
        "tests",
        operator,
        implementation_file,
    )
    benchmark_files = _find_suite_files(
        flaggems_repo,
        "benchmark",
        operator,
        implementation_file,
    )
    benchmark_shapes = _find_benchmark_shapes_from_yaml(
        flaggems_repo,
        operator,
        benchmark_files,
    )
    test_file = test_files[0] if test_files else _find_suite_file(
        flaggems_repo,
        "tests",
        operator,
        implementation_file,
    )
    benchmark_file = benchmark_files[0] if benchmark_files else _find_suite_file(
        flaggems_repo,
        "benchmark",
        operator,
        implementation_file,
    )
    implemented = operator in implemented_operators(flaggems_repo)
    if not test_file.exists():
        status = (
            "has an exported implementation but no operator pytest"
            if implemented
            else "has neither an exported implementation nor an operator pytest"
        )
        raise OperatorSourceNotFoundError(
            f"operator {operator!r} {status} "
            f"(repo={flaggems_repo!r}); refusing to invent correctness workloads"
        )

    return FlagGemsSourceInventory(
        operator=operator,
        repo=repo,
        labels=tuple(labels),
        benchmark_shapes=benchmark_shapes,
        operator_symbols=tuple(operator_symbols),
        implementation_file=implementation_file,
        public_signatures=_public_python_signatures(
            flaggems_repo,
            operator,
            operator_symbols,
            implementation_file,
        ),
        test_files=tuple(test_files),
        benchmark_files=tuple(benchmark_files),
        test_file=test_file,
        benchmark_file=benchmark_file,
        implemented=implemented,
        helper_context=_resolved_helper_context(repo, test_file, benchmark_file),
    )


@lru_cache(maxsize=None)
def _operator_catalog(flaggems_repo: str) -> list[dict[str, Any]]:
    """Return normalized operator-catalog entries, or an empty list."""

    catalog_path = Path(flaggems_repo) / "conf" / "operators.yaml"
    if not catalog_path.exists():
        return []
    try:
        data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = data.get("ops", data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _operator_entry(flaggems_repo: str, operator: str) -> dict[str, Any] | None:
    """Resolve the public catalog entry for an extractor operator name."""

    exact = next(
        (
            entry
            for entry in _operator_catalog(flaggems_repo)
            if entry.get("id") == operator or entry.get("name") == operator
        ),
        None,
    )
    if exact is not None:
        return exact
    # Leading underscores are sometimes omitted by the extractor-facing id,
    # but a trailing underscore distinguishes an in-place ATen operator from
    # its out-of-place counterpart.  Never collapse that ABI distinction.
    normalized = operator.lstrip("_")
    return next(
        (
            entry
            for entry in _operator_catalog(flaggems_repo)
            if str(entry.get("id") or entry.get("name") or "").lstrip("_")
            == normalized
        ),
        None,
    )


def _operator_symbols(flaggems_repo: str, operator: str) -> list[str]:
    """Return registered FlagGems symbols backing a public catalog id."""

    entry = _operator_entry(flaggems_repo, operator)
    symbols = entry.get("for") if entry else None
    if not isinstance(symbols, list):
        return [operator]
    normalized = [str(symbol) for symbol in symbols if str(symbol).strip()]
    return normalized or [operator]


@lru_cache(maxsize=None)
def _exported_operator_symbols(flaggems_repo: str) -> set[str]:
    """Read authoritative implementation symbols from ``ops`` and ``fused``."""

    package_root = checkout_profile(flaggems_repo).package_root(flaggems_repo)
    ops_init = package_root / "ops" / "__init__.py"
    if not ops_init.exists():
        raise FileNotFoundError(
            f"FlagGems repo not found or missing ops/__init__.py: {ops_init} "
            f"(flaggems_repo={flaggems_repo!r})"
        )
    exported: set[str] = set()
    for init_py in (ops_init, package_root / "fused" / "__init__.py"):
        if not init_py.exists():
            continue
        tree = ast.parse(init_py.read_text(encoding="utf-8"))
        package_exports: set[str] | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                getattr(target, "id", None) == "__all__" for target in node.targets
            ):
                package_exports = {
                    element.value
                    for element in node.value.elts
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                }
                break
        if package_exports is None:
            raise ValueError(f"No __all__ found in {init_py}")
        exported.update(package_exports)
    return exported


def implemented_operators(flaggems_repo: str) -> set[str]:
    """Return extractable public ids plus their directly exported symbols."""

    exported = _exported_operator_symbols(flaggems_repo)
    implemented = set(exported)
    for entry in _operator_catalog(flaggems_repo):
        public_id = entry.get("id") or entry.get("name")
        symbols = entry.get("for")
        if (
            isinstance(public_id, str)
            and isinstance(symbols, list)
            and any(str(symbol) in exported for symbol in symbols)
        ):
            implemented.add(public_id)
    implemented.update(_registered_operator_bindings(flaggems_repo))
    return implemented


@lru_cache(maxsize=None)
def _exported_operator_bindings(
    flaggems_repo: str,
) -> dict[str, tuple[Path, str]]:
    """Map each exported name to its source module and imported symbol."""

    repo = Path(flaggems_repo)
    bindings: dict[str, tuple[Path, str]] = {}
    for package in ("ops", "fused"):
        init_py = checkout_profile(repo).package_root(repo) / package / "__init__.py"
        if not init_py.exists():
            continue
        try:
            tree = ast.parse(init_py.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        prefix = f"{checkout_profile(repo).package}.{package}."
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith(prefix):
                continue
            relative = Path(*node.module[len(prefix) :].split(".")).with_suffix(".py")
            source = checkout_profile(repo).package_root(repo) / package / relative
            for alias in node.names:
                bindings[alias.asname or alias.name] = (source, alias.name)
    return bindings


@lru_cache(maxsize=None)
def _registered_operator_bindings(
    flaggems_repo: str,
) -> dict[str, tuple[tuple[str, Path, str], ...]]:
    """Resolve ATen registry aliases to their exported Python implementations.

    FlagGems can register a public name such as ``greater_equal.Tensor`` to a
    differently named implementation such as ``ge`` without exporting a
    ``greater_equal`` Python function.  The top-level registration table is the
    deterministic source of truth for that relationship.
    """

    init_py = checkout_profile(flaggems_repo).package_root(flaggems_repo) / "__init__.py"
    if not init_py.is_file():
        return {}
    try:
        tree = ast.parse(init_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    exports = _exported_operator_bindings(flaggems_repo)
    result: dict[str, list[tuple[str, Path, str]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "_FULL_CONFIG"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            continue
        for entry in node.value.elts:
            if (
                not isinstance(entry, (ast.Tuple, ast.List))
                or len(entry.elts) < 2
                or not isinstance(entry.elts[0], ast.Constant)
                or not isinstance(entry.elts[0].value, str)
                or not isinstance(entry.elts[1], ast.Name)
            ):
                continue
            registered_name = entry.elts[0].value
            implementation = exports.get(entry.elts[1].id)
            if implementation is None:
                continue
            source, implementation_name = implementation
            public_name = registered_name.split(".", 1)[0]
            item = (registered_name, source, implementation_name)
            if item not in result.setdefault(public_name, []):
                result[public_name].append(item)
    return {name: tuple(entries) for name, entries in result.items()}


def _preferred_registered_bindings(
    flaggems_repo: str,
    operator: str,
) -> tuple[tuple[str, Path, str], ...]:
    """Prefer the Tensor overload while retaining ambiguity for hard gating."""

    entries = _registered_operator_bindings(flaggems_repo).get(operator, ())
    tensor = tuple(
        entry
        for entry in entries
        if entry[0].partition(".")[2].split("_", 1)[0] == "Tensor"
    )
    direct = tuple(entry for entry in entries if entry[0] == operator)
    selected = tensor or direct or entries
    unique: list[tuple[str, Path, str]] = []
    seen: set[tuple[Path, str]] = set()
    for entry in selected:
        identity = (entry[1], entry[2])
        if identity not in seen:
            seen.add(identity)
            unique.append(entry)
    return tuple(unique)


@lru_cache(maxsize=None)
def _exported_operator_modules(flaggems_repo: str) -> dict[str, Path]:
    """Map exported symbols to their actual ``ops`` or ``fused`` modules."""

    return {
        public_name: source
        for public_name, (source, _) in _exported_operator_bindings(
            flaggems_repo
        ).items()
    }


def _signature_only_source(
    public_name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """Render a callable signature without copying its implementation body."""

    signature = ast.FunctionDef(
        name=public_name,
        args=function.args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=function.returns,
        type_comment=function.type_comment,
    )
    if hasattr(signature, "type_params"):
        signature.type_params = list(getattr(function, "type_params", []))
    return ast.unparse(ast.fix_missing_locations(signature)) + "\n"


def _public_python_signatures(
    flaggems_repo: str,
    operator: str,
    operator_symbols: list[str],
    implementation_file: Path,
) -> tuple[PublicPythonSignature, ...]:
    """Resolve signatures whose public export mapping is deterministic."""

    if not implementation_file.is_file():
        return ()
    try:
        tree = ast.parse(implementation_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ()
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    registered = _preferred_registered_bindings(flaggems_repo, operator)
    registered_signatures = [
        PublicPythonSignature(
            public_name=operator,
            implementation_name=implementation_name,
            source=_signature_only_source(operator, functions[implementation_name]),
        )
        for _, source, implementation_name in registered
        if source == implementation_file and implementation_name in functions
    ]
    if registered_signatures:
        return tuple(registered_signatures)

    expected_public_names = {
        name
        for name in (operator, *operator_symbols)
        if name.isidentifier()
    }
    signatures: list[PublicPythonSignature] = []
    for public_name, (source, implementation_name) in sorted(
        _exported_operator_bindings(flaggems_repo).items()
    ):
        if (
            source != implementation_file
            or public_name not in expected_public_names
        ):
            continue
        function = functions.get(implementation_name)
        if function is None:
            continue
        signatures.append(
            PublicPythonSignature(
                public_name=public_name,
                implementation_name=implementation_name,
                source=_signature_only_source(public_name, function),
            )
        )

    # Some small/fake repositories omit a re-export even though the direct public
    # function and file are unambiguous. Keep that deterministic fallback narrow.
    if not signatures and operator.isidentifier() and operator in functions:
        signatures.append(
            PublicPythonSignature(
                public_name=operator,
                implementation_name=operator,
                source=_signature_only_source(operator, functions[operator]),
            )
        )
    return tuple(signatures)


def _find_implementation_file(
    flaggems_repo: str,
    operator: str,
    symbols: list[str],
) -> Path:
    """Resolve an operator to its real implementation file."""

    repo = Path(flaggems_repo)
    package_root = checkout_profile(repo).package_root(repo)
    ops_dir = package_root / "ops"
    direct_names = [operator]
    for symbol in symbols:
        direct_names.append(symbol.split(".", 1)[0])

    # The package export is the public ABI authority.  Prefer it over a
    # same-named source file: FlagGems can retain an older ``foo_.py`` while
    # exporting ``foo_`` from ``foo.py``.
    exported_modules = _exported_operator_modules(flaggems_repo)
    for name in dict.fromkeys(direct_names):
        candidate = exported_modules.get(name)
        if candidate is not None and candidate.exists():
            return candidate

    registered = _preferred_registered_bindings(flaggems_repo, operator)
    if registered and registered[0][1].exists():
        return registered[0][1]

    for package in ("ops", "fused"):
        for name in dict.fromkeys(direct_names):
            candidate = package_root / package / f"{name}.py"
            if candidate.exists():
                return candidate

    return ops_dir / f"{operator}.py"


def _suite_name_variants(operator: str, implementation_file: Path) -> list[str]:
    """Return filename/marker variants used by related overload suites."""

    names: list[str] = []
    raw_names = [operator]
    if implementation_file.exists():
        raw_names.append(implementation_file.stem)
    for raw_name in raw_names:
        variants = [raw_name]
        if raw_name.startswith("_") and not raw_name.startswith("__"):
            variants.append(raw_name[1:])
        for name in tuple(variants):
            if name.endswith("_"):
                variants.append(name[:-1])
        stripped = raw_name.strip("_")
        if stripped:
            variants.append(stripped)
        names.extend(variants)
    return list(dict.fromkeys(names))


def _pytest_mark_names(source: str) -> set[str]:
    """Return exact ``pytest.mark.<name>`` attributes from Python source."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
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


def _has_module_level_pytest_skipif(path: Path) -> bool:
    """Detect an architecture/capability-gated supplementary pytest module.

    A secondary file such as ``test_rdna4_softmax.py`` can share the public
    operator marker while declaring one module-level ``pytestmark =
    pytest.mark.skipif(...)`` gate. Those files validate an optional installed
    implementation variant, not the canonical public operator semantics. Keep
    a gated file when it is the primary/only suite, but do not merge it into a
    generic primary suite merely because the marker name matches.
    """

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False

    def contains_skipif(node: ast.AST) -> bool:
        return any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "skipif"
            and isinstance(item.func.value, ast.Attribute)
            and item.func.value.attr == "mark"
            and isinstance(item.func.value.value, ast.Name)
            and item.func.value.value.id == "pytest"
            for item in ast.walk(node)
        )

    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "pytestmark"
        ):
            value = node.value
        if value is not None and contains_skipif(value):
            return True
    return False


@lru_cache(maxsize=None)
def _suite_marker_files(
    flaggems_repo: str, suite: str
) -> dict[str, tuple[Path, ...]]:
    """Index pytest markers once for deterministic multi-operator extraction."""

    index: dict[str, list[Path]] = {}
    suite_dir = Path(flaggems_repo) / suite
    if not suite_dir.is_dir():
        return {}
    for candidate in sorted(suite_dir.glob("test_*.py")):
        try:
            source = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        for marker in _pytest_mark_names(source):
            index.setdefault(marker, []).append(candidate)
    return {marker: tuple(paths) for marker, paths in index.items()}


def _find_suite_file(
    flaggems_repo: str,
    suite: str,
    operator: str,
    implementation_file: Path,
) -> Path:
    """Locate a pytest/benchmark suite shared by related overloads."""

    suite_dir = Path(flaggems_repo) / suite
    names = _suite_name_variants(operator, implementation_file)
    for name in names:
        candidate = suite_dir / f"test_{name}.py"
        if candidate.exists():
            return candidate
    marker_index = _suite_marker_files(flaggems_repo, suite)
    for name in names:
        marked = marker_index.get(name, ())
        if marked:
            return marked[0]
    return suite_dir / f"test_{operator}.py"


def _find_suite_files(
    flaggems_repo: str,
    suite: str,
    operator: str,
    implementation_file: Path,
) -> list[Path]:
    """Return the primary suite plus every file marked for the same operator."""

    primary = _find_suite_file(
        flaggems_repo,
        suite,
        operator,
        implementation_file,
    )
    files = [primary] if primary.exists() else []
    marker_names = {operator, operator.strip("_")}
    if operator.startswith("_") and not operator.startswith("__"):
        marker_names.add(operator[1:])
    marker_index = _suite_marker_files(flaggems_repo, suite)
    for marker in marker_names:
        for candidate in marker_index.get(marker, ()):
            if candidate != primary and _has_module_level_pytest_skipif(candidate):
                continue
            if candidate not in files:
                files.append(candidate)
    return files


def _top_level_source(path: Path, names: set[str]) -> list[str]:
    """Return exact top-level assignments/classes/functions for selected names."""

    if not path.is_file() or not names:
        return []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    snippets: list[str] = []
    for node in tree.body:
        node_names: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            node_names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        if node_names & names:
            segment = ast.get_source_segment(source, node)
            if segment:
                snippets.append(segment)
    return snippets


def _top_level_source_containing(path: Path, markers: set[str]) -> list[str]:
    """Return exact top-level statements containing benchmark policy markers."""

    if not path.is_file() or not markers:
        return []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    snippets: list[str] = []
    for node in tree.body:
        segment = ast.get_source_segment(source, node)
        if segment and any(marker in segment for marker in markers):
            snippets.append(segment)
    return snippets


def _referenced_helper_names(
    tree: ast.Module,
    qualifier: str,
    helper: Path,
) -> set[str]:
    """Resolve both module-qualified and ``from helper import name`` uses."""

    module_aliases = {qualifier}
    imported_names: set[str] = set()
    helper_name = helper.stem
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_tail = (node.module or "").rsplit(".", 1)[-1]
            if module_tail == helper_name:
                imported_names.update(
                    alias.name for alias in node.names if alias.name != "*"
                )
            elif node.module is None:
                module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == helper_name
                )
        elif isinstance(node, ast.Import):
            module_aliases.update(
                alias.asname or alias.name.rsplit(".", 1)[-1]
                for alias in node.names
                if alias.name.rsplit(".", 1)[-1] == helper_name
            )

    imported_names.update(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in module_aliases
    )
    return imported_names


def _unique_snippets(snippets: list[str]) -> list[str]:
    return list(dict.fromkeys(snippets))


def _resolved_helper_context(
    repo: Path,
    test_file: Path,
    benchmark_file: Path,
) -> str:
    """Inline helper symbols directly referenced by the selected source suites."""

    sources = (
        (test_file, "utils", repo / "tests" / "accuracy_utils.py"),
        (benchmark_file, "base", repo / "benchmark" / "base.py"),
        (benchmark_file, "consts", repo / "benchmark" / "consts.py"),
    )
    sections: list[str] = []
    for suite, qualifier, helper in sources:
        if not suite.is_file():
            continue
        try:
            tree = ast.parse(suite.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        names = _referenced_helper_names(tree, qualifier, helper)
        if qualifier == "base" and names:
            names.add("generate_tensor_input")
        snippets = _top_level_source(helper, names)
        if qualifier == "base" and names:
            snippets = _top_level_source_containing(
                helper,
                {"allow_tf32", "set_float32_matmul_precision"},
            ) + snippets
            snippets += _top_level_source(
                repo / "benchmark" / "consts.py",
                {"DEFAULT_SHAPES"},
            )
            if any("model_shapes(" in snippet for snippet in snippets):
                snippets += _top_level_source(
                    repo / "benchmark" / "consts.py",
                    {"model_shapes"},
                )
        if snippets:
            relative = helper.relative_to(repo).as_posix()
            sections.append(f"# {relative}\n" + "\n\n".join(snippets))
            if helper.name == "accuracy_utils.py" and any(
                f"{checkout_profile(repo).package}.testing.assert_close" in snippet for snippet in snippets
            ):
                testing = checkout_profile(repo).package_root(repo) / "testing" / "__init__.py"
                testing_snippets = _unique_snippets(
                    _top_level_source_containing(testing, {"RESOLUTION"})
                    + _top_level_source(testing, {"assert_close"})
                )
                if testing_snippets:
                    sections.append(
                        f"# src/{checkout_profile(repo).package}/testing/__init__.py\n"
                        + "\n\n".join(testing_snippets)
                    )
    if test_file.is_file():
        try:
            test_tree = ast.parse(test_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            test_tree = ast.Module(body=[], type_ignores=[])
        test_conftest = repo / "tests" / "conftest.py"
        conftest_names = _referenced_helper_names(
            test_tree, "conftest", test_conftest
        )
        conftest_snippets = _top_level_source(test_conftest, conftest_names)
        if conftest_snippets:
            sections.append(
                "# tests/conftest.py\n" + "\n\n".join(conftest_snippets)
            )
    if benchmark_file.is_file():
        config = _top_level_source(
            repo / "benchmark" / "conftest.py",
            {"BenchConfig"},
        )
        if config:
            sections.append(
                "# benchmark/conftest.py\n" + "\n\n".join(config)
            )
    return "\n\n".join(sections)


def _find_operator_labels(flaggems_repo: str, operator: str) -> list[str]:
    """Extract labels for an operator from conf/operators.yaml (best-effort)."""

    entry = _operator_entry(flaggems_repo, operator)
    if entry:
        labels = entry.get("labels", [])
        return labels if isinstance(labels, list) else []
    return []


def _find_benchmark_shapes_from_yaml(
    flaggems_repo: str,
    operator: str,
    benchmark_files: list[Path],
) -> str | None:
    """Resolve the effective core shape source in runtime precedence order.

    A benchmark subclass may override ``set_shapes`` and therefore completely
    bypass ``core_shapes.yaml``.  Resolve those simple source-defined lists
    first, then follow the instantiated benchmark class MRO through the YAML.
    """

    yaml_path = Path(flaggems_repo) / "benchmark" / "core_shapes.yaml"
    if not yaml_path.exists():
        return None
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        candidate_keys = [operator]
        instantiated_classes: list[str] = []
        local_classes: dict[str, ast.ClassDef] = {}
        module_constants: dict[str, Any] = {}
        for benchmark_file in benchmark_files:
            try:
                tree = ast.parse(benchmark_file.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    local_classes[node.name] = node
                    continue
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if value is None:
                    continue
                try:
                    resolved = ast.literal_eval(value)
                except (ValueError, TypeError):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        module_constants[target.id] = resolved
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                op_name = keywords.get("op_name")
                if not (
                    isinstance(op_name, ast.Constant)
                    and op_name.value == operator
                ):
                    continue
                if isinstance(node.func, ast.Attribute):
                    instantiated_classes.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    instantiated_classes.append(node.func.id)

        def class_bases(class_node: ast.ClassDef) -> list[str]:
            return [
                base.attr if isinstance(base, ast.Attribute) else base.id
                for base in class_node.bases
                if isinstance(base, (ast.Attribute, ast.Name))
            ]

        def inline_shapes(class_node: ast.ClassDef) -> Any | None:
            for method in class_node.body:
                if not (
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name == "set_shapes"
                ):
                    continue
                for statement in ast.walk(method):
                    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        continue
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    if not any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "shapes"
                        for target in targets
                    ):
                        continue
                    expression = statement.value
                    if expression is None:
                        continue
                    if isinstance(expression, ast.Name):
                        value = module_constants.get(expression.id)
                    else:
                        try:
                            value = ast.literal_eval(expression)
                        except (ValueError, TypeError):
                            value = None
                    if isinstance(value, (list, tuple)):
                        return value
            return None

        for class_name in dict.fromkeys(instantiated_classes):
            class_node = local_classes.get(class_name)
            if class_node is None:
                continue
            shapes = inline_shapes(class_node)
            if shapes is not None:
                return json.dumps(
                    {
                        "resolved_key": class_name,
                        "resolved_from": "benchmark set_shapes override",
                        "shapes": shapes,
                    },
                    indent=2,
                )

        base_classes: dict[str, list[str]] = {}
        base_path = Path(flaggems_repo) / "benchmark" / "base.py"
        try:
            base_tree = ast.parse(base_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            base_tree = ast.Module(body=[], type_ignores=[])
        for node in base_tree.body:
            if isinstance(node, ast.ClassDef):
                base_classes[node.name] = class_bases(node)

        def append_mro(class_name: str, seen: set[str]) -> None:
            if class_name in seen:
                return
            seen.add(class_name)
            candidate_keys.append(class_name)
            node = local_classes.get(class_name)
            bases = (
                class_bases(node)
                if node is not None
                else base_classes.get(class_name, [])
            )
            for base_name in bases:
                append_mro(base_name, seen)

        seen_classes: set[str] = set()
        for class_name in instantiated_classes:
            append_mro(class_name, seen_classes)
        for key in dict.fromkeys(candidate_keys):
            entry = data.get(key)
            if isinstance(entry, dict):
                return json.dumps(
                    {"resolved_key": key, **entry},
                    indent=2,
                )
    except Exception:
        pass
    return None
