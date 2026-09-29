"""Hard gates derived from resolved FlagGems benchmark source."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from kernelgen.agents.extractor.flaggems.models import ExtractorResult, Workload


def benchmark_has_blas_column_major_control(
    benchmark_files: Iterable[Path],
) -> bool:
    """Return whether a resolved benchmark exposes BlasBenchmark's B-layout branch."""

    for path in benchmark_files:
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        has_layout_control = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(argument.arg == "b_column_major" for argument in node.args.args)
            for node in ast.walk(module)
        )
        if not has_layout_control:
            continue
        if any(
            isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "BlasBenchmark"
                )
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "BlasBenchmark"
                )
            )
            for node in ast.walk(module)
        ):
            return True
    return False


@dataclass(frozen=True)
class CoreAddmmTimingProfile:
    """Source-resolved ``test-op.sh --level core`` sequence for ``addmm_``."""

    dtypes: tuple[str, ...]
    shapes: tuple[tuple[int, int, int, int], ...]

    @property
    def expected_count(self) -> int:
        return len(self.dtypes) * len(self.shapes)

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "runner": "tools/test-op.sh",
            "pytest_level": "core",
            "dtypes": list(self.dtypes),
            "source_shapes": [list(shape) for shape in self.shapes],
            "iteration_order": "dtype-major, then source shape order",
            "b_column_major": False,
            "expected_workloads": self.expected_count,
        }


def _dtype_constants(path: Path) -> dict[str, tuple[str, ...]]:
    """Resolve simple torch dtype lists from benchmark/consts.py."""

    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    constants: dict[str, tuple[str, ...]] = {}

    def resolve(expression: ast.expr) -> tuple[str, ...] | None:
        if isinstance(expression, (ast.List, ast.Tuple)):
            values: list[str] = []
            for element in expression.elts:
                if not (
                    isinstance(element, ast.Attribute)
                    and isinstance(element.value, ast.Name)
                    and element.value.id == "torch"
                ):
                    return None
                values.append(element.attr)
            return tuple(values)
        if isinstance(expression, ast.Name):
            return constants.get(expression.id)
        return None

    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        values = resolve(node.value)
        if values is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = values
    return constants


def _resolve_dtype_expression(
    expression: ast.expr,
    constants: dict[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    if isinstance(expression, (ast.List, ast.Tuple)):
        values: list[str] = []
        for element in expression.elts:
            if not isinstance(element, ast.Attribute):
                return None
            values.append(element.attr)
        return tuple(values)
    if isinstance(expression, ast.Name):
        return constants.get(expression.id)
    if isinstance(expression, ast.Attribute):
        return constants.get(expression.attr)
    return None


def _resolved_addmm_dtypes(
    operator: str,
    benchmark_files: Iterable[Path],
) -> tuple[str, ...] | None:
    for path in benchmark_files:
        constants = _dtype_constants(path.parent / "consts.py")
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            callable_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id if isinstance(node.func, ast.Name) else None
            )
            if callable_name != "BlasBenchmark":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            op_name = keywords.get("op_name")
            if not (
                isinstance(op_name, ast.Constant)
                and op_name.value == operator
            ):
                continue
            dtype_expression = keywords.get("dtypes")
            if dtype_expression is None:
                return constants.get("FLOAT_DTYPES")
            return _resolve_dtype_expression(dtype_expression, constants)
    return None


def resolve_core_addmm_timing_profile(
    operator: str,
    benchmark_files: Iterable[Path],
    benchmark_shapes: str | None,
) -> CoreAddmmTimingProfile | None:
    """Resolve the exact core sequence when the source is the addmm_ BLAS suite."""

    if operator != "addmm_" or not benchmark_shapes:
        return None
    try:
        payload = json.loads(benchmark_shapes)
    except (TypeError, json.JSONDecodeError):
        return None
    raw_shapes = payload.get("shapes") if isinstance(payload, dict) else None
    if not isinstance(raw_shapes, list):
        return None
    shapes: list[tuple[int, int, int, int]] = []
    for raw_shape in raw_shapes:
        if not (
            isinstance(raw_shape, list)
            and len(raw_shape) == 4
            and all(isinstance(value, int) for value in raw_shape)
        ):
            return None
        shapes.append(tuple(raw_shape))
    dtypes = _resolved_addmm_dtypes(operator, benchmark_files)
    if not shapes or not dtypes:
        return None
    return CoreAddmmTimingProfile(dtypes=dtypes, shapes=tuple(shapes))


def _literal_scalar_values(
    expression: ast.expr,
    constants: dict[str, tuple[Any, ...]],
) -> tuple[Any, ...] | None:
    if isinstance(expression, (ast.Name, ast.Attribute)):
        name = expression.id if isinstance(expression, ast.Name) else expression.attr
        return constants.get(name)
    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, (list, tuple)) or not all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    ):
        return None
    return tuple(value)


def _scalar_constants(source: str) -> dict[str, tuple[Any, ...]]:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return {}
    constants: dict[str, tuple[Any, ...]] = {}
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        resolved = _literal_scalar_values(value, constants)
        if resolved is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = resolved
    return constants


def pytest_keyword_scalar_requirements(
    operator: str,
    test_files: Iterable[Path],
    helper_context: str,
) -> dict[str, tuple[Any, ...]]:
    """Resolve finite pytest scalar parametrizations reaching public keywords."""

    constants = _scalar_constants(helper_context)
    requirements: dict[str, set[Any]] = {}
    for path in test_files:
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        file_constants = dict(constants)
        file_constants.update(_scalar_constants(source))
        for function in (
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            parametrized: dict[str, tuple[Any, ...]] = {}
            for decorator in function.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "parametrize"
                    and len(decorator.args) >= 2
                    and isinstance(decorator.args[0], ast.Constant)
                    and isinstance(decorator.args[0].value, str)
                    and "," not in decorator.args[0].value
                ):
                    continue
                values = _literal_scalar_values(decorator.args[1], file_constants)
                if values is not None:
                    parametrized[decorator.args[0].value.strip()] = values
            if not parametrized:
                continue

            aliases: dict[str, str] = {}
            for node in ast.walk(function):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if not isinstance(value, ast.Name):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = value.id

            def root_name(name: str) -> str:
                seen: set[str] = set()
                while name in aliases and name not in seen:
                    seen.add(name)
                    name = aliases[name]
                return name

            for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
                callable_name = None
                if isinstance(call.func, ast.Name):
                    callable_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    callable_name = call.func.attr
                if callable_name != operator:
                    continue
                for keyword in call.keywords:
                    if keyword.arg is None or not isinstance(keyword.value, ast.Name):
                        continue
                    source_name = root_name(keyword.value.id)
                    if source_name in parametrized:
                        requirements.setdefault(keyword.arg, set()).update(
                            parametrized[source_name]
                        )
    return {
        name: tuple(sorted(values, key=repr))
        for name, values in requirements.items()
    }


def validate_keyword_scalar_coverage(
    result: ExtractorResult,
    requirements: dict[str, tuple[Any, ...]],
) -> None:
    """Require every resolved pytest keyword value in correctness workloads."""

    observed: dict[str, set[Any]] = {name: set() for name in requirements}
    for workload in result.correctness_workloads:
        try:
            call = ast.parse(workload.call, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(call, ast.Call):
            continue
        for keyword in call.keywords:
            if (
                keyword.arg not in observed
                or not isinstance(keyword.value, ast.Name)
                or keyword.value.id not in workload.inputs
            ):
                continue
            spec = workload.inputs[keyword.value.id]
            if spec.type in {"scalar", "literal"}:
                observed[keyword.arg].add(spec.value)

    missing = {
        name: sorted(set(values) - observed[name], key=repr)
        for name, values in requirements.items()
        if set(values) - observed[name]
    }
    if missing:
        raise ValueError(
            f"{result.definition.name}: correctness workloads omit source pytest "
            f"parametrized keyword values {missing}"
        )


def pytest_requires_equal_nan(
    operator: str,
    test_files: Iterable[Path],
) -> bool:
    """Return whether an operator pytest mode explicitly asserts equal NaNs."""

    for path in test_files:
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for function in (
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
            calls_operator = any(
                (
                    isinstance(call.func, ast.Name)
                    and call.func.id == operator
                )
                or (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == operator
                )
                for call in calls
            )
            if not calls_operator:
                continue
            if any(
                any(
                    keyword.arg == "equal_nan"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in call.keywords
                )
                for call in calls
            ):
                return True
    return False


def validate_equal_nan_coverage(result: ExtractorResult, required: bool) -> None:
    """Require the Workload comparison policy used by its source pytest."""

    if not required:
        return
    missing = [
        workload.name
        for workload in result.correctness_workloads
        if workload.tolerance is None or not workload.tolerance.equal_nan
    ]
    if missing:
        raise ValueError(
            f"{result.definition.name}: source pytest explicitly uses "
            f"equal_nan=True; correctness workloads missing "
            f"tolerance.equal_nan=true: {missing}"
        )


def _is_column_major(workload: Workload) -> bool:
    return any(
        spec.generator_params is not None
        and spec.generator_params.get("column_major") is True
        for spec in workload.inputs.values()
    )


def validate_core_blas_layout(result: ExtractorResult) -> None:
    """Reject comprehensive-only column-major cases from the core timing set."""

    if any(_is_column_major(workload) for workload in result.timing_workloads):
        raise ValueError(
            f"{result.definition.name}: tools/test-op.sh runs benchmarks with "
            "--level core, where BlasBenchmark does not enter "
            "b_column_major=True; timing workloads must not contain "
            "generator_params.column_major=true"
        )


def _validate_addmm_timing_case(
    result: ExtractorResult,
    workload: Workload,
    index: int,
    dtype: str,
    source_shape: tuple[int, int, int, int],
) -> None:
    try:
        expression = ast.parse(workload.call, mode="eval").body
    except SyntaxError as exc:  # The model normally rejects this first.
        raise ValueError(
            f"{result.definition.name}: invalid timing call at core index {index}"
        ) from exc
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "addmm_"
        and len(expression.args) == 3
        and not expression.keywords
        and all(isinstance(argument, ast.Name) for argument in expression.args)
    ):
        raise ValueError(
            f"{result.definition.name}: test-op.sh --level core timing index {index} "
            "must call addmm_(self, mat1, mat2) positionally and omit alpha/beta"
        )

    _, m, n, k = source_shape
    expected_shapes = ((m, n), (m, k), (k, n))
    for argument, expected_shape in zip(expression.args, expected_shapes):
        assert isinstance(argument, ast.Name)
        spec = workload.inputs.get(argument.id)
        if not (
            spec is not None
            and spec.type == "random"
            and spec.shape == list(expected_shape)
            and spec.dtype == dtype
        ):
            actual = None if spec is None else spec.model_dump(exclude_none=True)
            raise ValueError(
                f"{result.definition.name}: test-op.sh --level core timing index "
                f"{index} input {argument.id!r} must be random "
                f"shape={list(expected_shape)}, dtype={dtype}; got {actual}"
            )


def validate_core_addmm_timing(
    result: ExtractorResult,
    profile: CoreAddmmTimingProfile,
) -> None:
    """Require addmm_ timing to preserve the exact source-derived core sequence."""

    workloads = result.timing_workloads
    if len(workloads) != profile.expected_count:
        raise ValueError(
            f"{result.definition.name}: test-op.sh --level core requires exactly "
            f"{profile.expected_count} timing workloads in source order; got "
            f"{len(workloads)}. Do not sample, deduplicate, or append "
            "comprehensive-only cases."
        )

    index = 0
    for dtype in profile.dtypes:
        for source_shape in profile.shapes:
            _validate_addmm_timing_case(
                result,
                workloads[index],
                index,
                dtype,
                source_shape,
            )
            index += 1
