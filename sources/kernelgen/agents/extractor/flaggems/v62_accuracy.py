"""Accuracy-source coverage rules for direct FlagGems v6.2 extraction.

This module owns the boundary between pytest source and emitted correctness
workloads.  It intentionally does not know about oracle implementation details
or catalog persistence.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Protocol

from kernelgen.agents.extractor.flaggems.source_inventory import (
    FlagGemsSourceInventory,
)


AUTOGRAD_CONTRACT_REASON = "unrepresentable: autograd contract"
EXPECTED_EXCEPTION_REASON = "expected exception only"


class _NamedWorkload(Protocol):
    name: str
    inputs: dict[str, Any]


@dataclass(frozen=True)
class AccuracySource:
    """One marked pytest function and its v6.2 coverage disposition."""

    identity: str
    function: str
    exclusion_reason: str | None = None
    uncovered_contracts: tuple[str, ...] = ()


@dataclass(frozen=True)
class AccuracyCoveragePlan:
    """Source functions that must be represented or explicitly excluded."""

    sources: tuple[AccuracySource, ...]
    all_test_functions: frozenset[str]

    @property
    def marked_functions(self) -> frozenset[str]:
        return frozenset(source.function for source in self.sources)

    @property
    def identities(self) -> frozenset[str]:
        return frozenset(source.identity for source in self.sources)

    @property
    def required_identities(self) -> frozenset[str]:
        return frozenset(
            source.identity
            for source in self.sources
            if source.exclusion_reason is None
        )

    @property
    def excluded_identities(self) -> dict[str, str]:
        return {
            source.identity: source.exclusion_reason
            for source in self.sources
            if source.exclusion_reason is not None
        }

    @property
    def partial_identities(self) -> dict[str, tuple[str, ...]]:
        return {
            source.identity: source.uncovered_contracts
            for source in self.sources
            if source.uncovered_contracts
        }


def _attribute_path(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent is not None:
            return (*parent, node.attr)
    return None


def _is_operator_marked(function: ast.FunctionDef, operator: str) -> bool:
    return any(
        _attribute_path(
            decorator.func if isinstance(decorator, ast.Call) else decorator
        )
        == ("pytest", "mark", operator)
        for decorator in function.decorator_list
    )


def _is_autograd_call(node: ast.Call) -> bool:
    path = _attribute_path(node.func)
    return bool(
        path == ("torch", "autograd", "grad")
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "backward")
    )


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {
            name
            for element in target.elts
            for name in _assigned_names(element)
        }
    return set()


def _autograd_result_names(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and _attribute_path(value.func) == ("torch", "autograd", "grad")
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(_assigned_names(target))
    return names


def _is_assertion_call(node: ast.Call) -> bool:
    path = _attribute_path(node.func)
    if path is None or path == ("pytest", "raises"):
        return False
    name = path[-1]
    return name.startswith("assert") or name.startswith("gems_assert")


def _assertion_expressions(function: ast.FunctionDef) -> list[ast.expr]:
    expressions: list[ast.expr] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assert):
            expressions.append(node.test)
        elif isinstance(node, ast.Call) and _is_assertion_call(node):
            # Accuracy helpers compare their first two values; later arguments
            # are commonly dtype or tolerance metadata.
            expressions.extend(node.args[:2])
    return expressions


def _is_gradient_expression(node: ast.expr, gradient_names: set[str]) -> bool:
    return any(
        (isinstance(child, ast.Attribute) and child.attr == "grad")
        or (isinstance(child, ast.Name) and child.id in gradient_names)
        for child in ast.walk(node)
    )


def _classify_marked_test(
    identity: str,
    function: ast.FunctionDef,
    *,
    public_backward: bool,
) -> AccuracySource:
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    has_autograd = any(_is_autograd_call(call) for call in calls)
    has_expected_exception = any(
        _attribute_path(call.func) == ("pytest", "raises") for call in calls
    )
    assertions = _assertion_expressions(function)
    gradient_names = _autograd_result_names(function)
    gradient_assertions = [
        expression
        for expression in assertions
        if _is_gradient_expression(expression, gradient_names)
    ]
    forward_assertions = [
        expression
        for expression in assertions
        if not _is_gradient_expression(expression, gradient_names)
    ]

    if (
        not public_backward
        and has_autograd
        and gradient_assertions
        and not forward_assertions
    ):
        return AccuracySource(
            identity=identity,
            function=function.name,
            exclusion_reason=AUTOGRAD_CONTRACT_REASON,
        )
    if has_expected_exception and not assertions:
        return AccuracySource(
            identity=identity,
            function=function.name,
            exclusion_reason=EXPECTED_EXCEPTION_REASON,
        )
    uncovered = (
        ("autograd contract",)
        if (
            not public_backward
            and has_autograd
            and gradient_assertions
            and forward_assertions
        )
        else ()
    )
    return AccuracySource(
        identity=identity,
        function=function.name,
        uncovered_contracts=uncovered,
    )


def accuracy_coverage_plan(
    inventory: FlagGemsSourceInventory,
    operator: str,
) -> AccuracyCoveragePlan:
    """Analyze every explicitly marked accuracy pytest function once."""

    sources: list[AccuracySource] = []
    all_tests: set[str] = set()
    for path in inventory.test_files:
        repo = getattr(inventory, "repo", path.parent)
        try:
            relative = path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            relative = path.name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ):
            all_tests.add(function.name)
            if _is_operator_marked(function, operator):
                sources.append(
                    _classify_marked_test(
                        f"{relative}::{function.name}",
                        function,
                        public_backward="_backward" in operator,
                    )
                )
    return AccuracyCoveragePlan(
        sources=tuple(sorted(sources, key=lambda source: source.identity)),
        all_test_functions=frozenset(all_tests),
    )


def accuracy_test_scope(
    inventory: FlagGemsSourceInventory,
    operator: str,
) -> tuple[set[str], set[str]]:
    plan = accuracy_coverage_plan(inventory, operator)
    return set(plan.marked_functions), set(plan.all_test_functions)


def accuracy_test_identities(
    inventory: FlagGemsSourceInventory,
    operator: str,
) -> set[str]:
    return set(accuracy_coverage_plan(inventory, operator).identities)


def duplicate_accuracy_test_identities(identities: set[str]) -> set[str]:
    by_function: dict[str, set[str]] = {}
    for identity in identities:
        by_function.setdefault(identity.rsplit("::", 1)[-1], set()).add(identity)
    return {
        identity
        for same_name in by_function.values()
        if len(same_name) > 1
        for identity in same_name
    }


def _represented_source_identities(
    workloads: list[_NamedWorkload],
    identities: set[str],
    *,
    operator: str | None = None,
) -> set[str]:
    by_function: dict[str, set[str]] = {}
    for identity in identities:
        by_function.setdefault(identity.rsplit("::", 1)[-1], set()).add(identity)

    represented: set[str] = set()
    for workload in workloads:
        segments = {
            segment.split("[", 1)[0]
            for component in workload.name.split("::")
            for segment in component.split("/")
            if segment
        }
        for identity in identities:
            path, function = identity.rsplit("::", 1)
            path_candidates = {path, path.rsplit("/", 1)[-1]}
            has_path = any(candidate in workload.name for candidate in path_candidates)
            same_name_sources = by_function[function]
            if function in segments and (has_path or len(same_name_sources) == 1):
                represented.add(identity)
    # Catalogs emitted before source identities became mandatory sometimes use
    # only the operator name.  This is unambiguous when exactly one marked
    # source exists, so retain read/repersist compatibility without weakening
    # multi-source coverage checks.
    if (
        not represented
        and len(identities) == 1
        and operator is not None
        and any(operator in workload.name for workload in workloads)
    ):
        represented.update(identities)
    return represented


def validate_accuracy_workload_scope(
    operator: str,
    workloads: list[_NamedWorkload],
    *,
    allowed: set[str],
    all_tests: set[str],
    required_identities: set[str] | None = None,
    excluded_identities: dict[str, str] | None = None,
) -> None:
    if not allowed:
        return
    invalid: list[tuple[str, str]] = []
    for workload in workloads:
        referenced = set()
        case = workload.inputs.get("case")
        if isinstance(case, dict):
            referenced.update(
                case[key]
                for key in ("test", "test_fn")
                if isinstance(case.get(key), str)
            )
        referenced.update(
            candidate
            for component in workload.name.split("::")
            for segment in component.split("/")
            for candidate in (segment.split("[", 1)[0],)
            if candidate in all_tests
        )
        invalid.extend((workload.name, name) for name in referenced - allowed)
    if invalid:
        examples = [f"{workload}:{name}" for workload, name in invalid[:5]]
        raise ValueError(
            f"{operator}: correctness workloads may only come from pytest "
            f"functions marked pytest.mark.{operator}; invalid sibling tests "
            f"{examples}, allowed={sorted(allowed)}"
        )

    required_identities = required_identities or set()
    excluded_identities = excluded_identities or {}
    identities = required_identities | set(excluded_identities)
    represented = _represented_source_identities(
        workloads, identities, operator=operator
    )
    invalid_exclusions = represented & set(excluded_identities)
    if invalid_exclusions:
        details = {
            identity: excluded_identities[identity]
            for identity in sorted(invalid_exclusions)
        }
        raise ValueError(
            f"{operator}: correctness workloads include excluded pytest "
            f"sources {details}"
        )
    missing = required_identities - represented
    if missing:
        raise ValueError(
            f"{operator}: correctness workloads do not represent every "
            f"required marked pytest source; missing {sorted(missing)}"
        )


def accuracy_coverage_report(
    operator: str,
    plan: AccuracyCoveragePlan,
    workloads: list[_NamedWorkload],
) -> dict[str, Any]:
    represented = _represented_source_identities(
        workloads, set(plan.identities), operator=operator
    )
    return {
        "schema_version": "kernelgen.flaggems-v6.2-accuracy-coverage/v1",
        "operator": operator,
        "required_sources": sorted(plan.required_identities),
        "represented_sources": sorted(
            represented & set(plan.required_identities)
        ),
        "excluded_sources": [
            {"identity": identity, "reason": reason}
            for identity, reason in sorted(plan.excluded_identities.items())
        ],
        "partial_sources": [
            {"identity": identity, "uncovered": list(uncovered)}
            for identity, uncovered in sorted(plan.partial_identities.items())
        ],
    }


__all__ = [
    "AUTOGRAD_CONTRACT_REASON",
    "AccuracyCoveragePlan",
    "AccuracySource",
    "accuracy_coverage_plan",
    "accuracy_coverage_report",
    "accuracy_test_identities",
    "accuracy_test_scope",
    "duplicate_accuracy_test_identities",
    "validate_accuracy_workload_scope",
]
