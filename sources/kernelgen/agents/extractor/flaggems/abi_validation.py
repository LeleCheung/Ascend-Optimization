"""Static validation helpers for V6 reference source and public ABI."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernelgen.agents.extractor.flaggems.models import Parameter


FRAMEWORK_TYPES = {
    "dtype",
    "device",
    "layout",
    "memory_format",
    "Generator",
    "Optional[dtype]",
    "Optional[device]",
    "Optional[layout]",
    "Optional[memory_format]",
    "Optional[Generator]",
}


def finite_json(value: Any, path: str = "value") -> None:
    """Reject Python's non-standard JSON NaN/Infinity extensions recursively."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"{path} must be finite JSON; encode nan/inf as a custom recipe"
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{path}.{key}")


def normalize_ast_default(node: ast.expr, parameter_type: str | None) -> Any:
    """Turn a trusted Python default into its V6 JSON token."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            finite_json(node.value, "default")
            return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = normalize_ast_default(node.operand, parameter_type)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [normalize_ast_default(element, None) for element in node.elts]
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                break
            result[key.value] = normalize_ast_default(value, None)
        else:
            return result
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and parameter_type in FRAMEWORK_TYPES
    ):
        return node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "device"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and parameter_type in {"device", "Optional[device]"}
    ):
        return node.args[0].value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in {"inf", "-inf", "nan"}
    ):
        return node.args[0].value
    raise ValueError(
        "run() default cannot be normalized for v6: "
        f"{ast.unparse(node)!r}"
    )


def same_default(left: Any, right: Any) -> bool:
    """Compare normalized defaults without treating bool/int as equal."""

    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same_default(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same_default(left[key], right[key]) for key in left
        )
    return left == right


def top_level_functions(module: ast.Module) -> dict[str, ast.FunctionDef]:
    functions: dict[str, ast.FunctionDef] = {}
    for node in module.body:
        if isinstance(node, ast.AsyncFunctionDef):
            raise ValueError(f"top-level hook {node.name}() must not be async")
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    return functions


def function_parameter_rows(
    function: ast.FunctionDef,
) -> list[tuple[str, str, bool | None, ast.expr | None]]:
    """Return Python parameter rows in public signature order."""

    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    rows: list[tuple[str, str, bool | None, ast.expr | None]] = []
    positional_only_count = len(arguments.posonlyargs)
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        kind = (
            "positional_only"
            if index < positional_only_count
            else "positional_or_keyword"
        )
        rows.append((argument.arg, kind, default is None, default))
    if arguments.vararg is not None:
        rows.append((arguments.vararg.arg, "var_positional", None, None))
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        rows.append((argument.arg, "keyword_only", default is None, default))
    if arguments.kwarg is not None:
        rows.append((arguments.kwarg.arg, "var_keyword", None, None))
    return rows


def validate_hook_signature(
    definition_name: str,
    function: ast.FunctionDef,
    expected: tuple[str, ...],
) -> None:
    rows = function_parameter_rows(function)
    actual = tuple(
        name
        for name, kind, required, _ in rows
        if kind == "positional_or_keyword" and required
    )
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(
            f"{definition_name}: {function.name}() must have exact signature "
            f"({', '.join(expected)})"
        )


def validate_gen_inputs_randomness(
    definition_name: str,
    function: ast.FunctionDef,
    *,
    allow_target_device: bool = False,
) -> None:
    """Validate random allocation portability in an input-builder function."""

    random_factories = {
        "bernoulli",
        "multinomial",
        "normal",
        "rand",
        "randint",
        "randn",
        "randperm",
    }

    alias_values: dict[str, list[ast.expr]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    alias_values.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                alias_values.setdefault(node.target.id, []).append(node.value)

    def generator_device(value: ast.expr) -> str | None:
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "torch"
            and value.func.attr == "Generator"
        ):
            return None
        keyword = next(
            (item for item in value.keywords if item.arg == "device"), None
        )
        if keyword is None:
            return None
        if isinstance(keyword.value, ast.Name) and keyword.value.id == "device":
            return "target"
        if (
            isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "cpu"
        ):
            return "cpu"
        return None

    portable_generators = {
        name
        for name, values in alias_values.items()
        if values and all(generator_device(value) is not None for value in values)
    }

    def portable_device_value(value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Name)
            and value.id == "device"
        ) or (
            isinstance(value, ast.Constant)
            and value.value == "cpu"
        ) or (
            isinstance(value, ast.Attribute)
            and value.attr == "device"
            and isinstance(value.value, ast.Name)
            and value.value.id in portable_generators
        )

    portable_device_aliases = {
        name
        for name, values in alias_values.items()
        if values and all(portable_device_value(value) for value in values)
    }
    factory_device_aliases: set[str] = set()
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr in random_factories
        ):
            continue
        device_keyword = next(
            (keyword for keyword in node.keywords if keyword.arg == "device"),
            None,
        )
        if device_keyword is None:
            continue
        device_value = device_keyword.value
        if isinstance(device_value, ast.Constant) and device_value.value == "cpu":
            continue
        if isinstance(device_value, ast.Name) and device_value.id == "device":
            if allow_target_device:
                continue
            raise ValueError(
                f"{definition_name}: gen_inputs() random factory torch."
                f"{node.func.attr} must generate its base value on CPU and then "
                "move the result to device"
            )
        if (
            allow_target_device
            and isinstance(device_value, ast.Name)
            and device_value.id in portable_device_aliases
        ):
            factory_device_aliases.add(device_value.id)
            continue
        raise ValueError(
            f"{definition_name}: {function.name}() random factory torch."
            f"{node.func.attr} must allocate on the gen_inputs device argument "
            "or on CPU in a guarded fallback; hard-coded target devices are "
            "not portable"
        )

    def random_devices(nodes: list[ast.stmt]) -> set[str]:
        devices: set[str] = set()
        for statement in nodes:
            for call in ast.walk(statement):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "torch"
                    and call.func.attr in random_factories
                ):
                    continue
                keyword = next(
                    (item for item in call.keywords if item.arg == "device"),
                    None,
                )
                if keyword is None:
                    devices.add("implicit")
                elif (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "cpu"
                ):
                    devices.add("cpu")
                elif (
                    isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "device"
                ):
                    devices.add("target")
                elif (
                    isinstance(keyword.value, ast.Name)
                    and keyword.value.id in portable_device_aliases
                ):
                    for value in alias_values[keyword.value.id]:
                        if isinstance(value, ast.Name):
                            devices.add("target")
                        elif isinstance(value, ast.Constant):
                            devices.add("cpu")
                        elif isinstance(value, ast.Attribute):
                            for generator_value in alias_values[value.value.id]:
                                kind = generator_device(generator_value)
                                if kind is not None:
                                    devices.add(kind)
        return devices

    if not allow_target_device:
        return

    for node in ast.walk(function):
        if not isinstance(node, ast.Try):
            continue
        body_devices = random_devices(node.body)
        fallback_devices = set()
        for handler in node.handlers:
            fallback_devices.update(random_devices(handler.body))
        if "target" in body_devices and ({"cpu", "implicit"} & fallback_devices):
            raise ValueError(
                f"{definition_name}: {function.name}() must not catch target "
                "random-allocation failures and retry on CPU; guard only target "
                "Generator construction"
            )

        def generator_assignments(
            statements: list[ast.stmt], expected_device: str
        ) -> set[str]:
            names = set()
            for statement in statements:
                for child in ast.walk(statement):
                    if not (
                        isinstance(child, ast.Assign)
                        and isinstance(child.value, ast.Call)
                        and isinstance(child.value.func, ast.Attribute)
                        and isinstance(child.value.func.value, ast.Name)
                        and child.value.func.value.id == "torch"
                        and child.value.func.attr == "Generator"
                    ):
                        continue
                    device_keyword = next(
                        (
                            keyword
                            for keyword in child.value.keywords
                            if keyword.arg == "device"
                        ),
                        None,
                    )
                    matches = (
                        expected_device == "target"
                        and device_keyword is not None
                        and isinstance(device_keyword.value, ast.Name)
                        and device_keyword.value.id == "device"
                    ) or (
                        expected_device == "cpu"
                        and device_keyword is not None
                        and isinstance(device_keyword.value, ast.Constant)
                        and device_keyword.value.value == "cpu"
                    )
                    if matches:
                        names.update(
                            target.id
                            for target in child.targets
                            if isinstance(target, ast.Name)
                        )
            return names

        target_generators = generator_assignments(node.body, "target")
        for handler in node.handlers:
            fallback_generators = generator_assignments(handler.body, "cpu")
            if not target_generators & fallback_generators:
                continue
            if any(isinstance(child, ast.Return) for child in ast.walk(handler)):
                continue
            handler_aliases = {
                target.id
                for statement in handler.body
                for child in ast.walk(statement)
                if isinstance(child, ast.Assign)
                and isinstance(child.value, ast.Constant)
                and child.value.value == "cpu"
                for target in child.targets
                if isinstance(target, ast.Name)
            }
            body_aliases = {
                target.id
                for statement in node.body
                for child in ast.walk(statement)
                if isinstance(child, ast.Assign)
                and isinstance(child.value, ast.Name)
                and child.value.id == "device"
                for target in child.targets
                if isinstance(target, ast.Name)
            }
            if handler_aliases & body_aliases:
                continue
            generator_device_aliases = {
                name
                for name, values in alias_values.items()
                if name in factory_device_aliases
                and values
                and all(
                    isinstance(value, ast.Attribute)
                    and value.attr == "device"
                    and isinstance(value.value, ast.Name)
                    and value.value.id in target_generators & fallback_generators
                    for value in values
                )
            }
            if generator_device_aliases:
                continue
            raise ValueError(
                f"{definition_name}: {function.name}() CPU Generator fallback "
                "must either build on CPU and return the moved tensor, or set "
                "a shared factory-device alias so generator and factory devices "
                "always match"
            )


def validate_hook_context_access(
    definition_name: str,
    function: ast.FunctionDef,
) -> None:
    """Reject the obsolete nested-input view of the flat V6 hook context."""

    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript):
            continue
        if not (
            isinstance(node.slice, ast.Constant)
            and node.slice.value == "inputs"
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ctx"
            and isinstance(node.value.slice, ast.Constant)
            and node.value.slice.value in {"specs", "values"}
        ):
            continue
        raise ValueError(
            f"{definition_name}: {function.name}() uses obsolete "
            f"ctx[{node.value.slice.value!r}]['inputs']; V6 exposes a flat "
            f"ctx[{node.value.slice.value!r}] mapping keyed directly by Workload "
            "input name"
        )


def validate_run_abi(
    definition_name: str,
    parameters: list[Parameter],
    function: ast.FunctionDef,
    source_label: str,
) -> None:
    rows = function_parameter_rows(function)
    if len(rows) != len(parameters):
        raise ValueError(
            f"{definition_name}: {source_label} run() has {len(rows)} parameters, "
            f"Definition.parameters has {len(parameters)}"
        )
    for expected, (name, kind, required, default_node) in zip(parameters, rows):
        if expected.name != name or expected.kind != kind:
            raise ValueError(
                f"{definition_name}: {source_label} run() ABI differs at "
                f"{expected.name!r}: expected {expected.kind} {expected.name}, "
                f"got {kind} {name}"
            )
        if expected.kind in {"var_positional", "var_keyword"}:
            continue
        if expected.required is not required:
            raise ValueError(
                f"{definition_name}: {source_label} run() required/default status "
                f"differs for {expected.name}"
            )
        if not expected.required:
            assert default_node is not None
            actual_default = normalize_ast_default(default_node, expected.type)
            if not same_default(expected.default, actual_default):
                raise ValueError(
                    f"{definition_name}: {source_label} run() default differs for "
                    f"{expected.name}: Definition={expected.default!r}, "
                    f"run={actual_default!r}"
                )


def parse_reference(
    definition_name: str,
    source: str,
    label: str,
) -> tuple[ast.Module, dict[str, ast.FunctionDef]]:
    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(
            f"{definition_name}: invalid {label} source: {error}"
        ) from error
    if any(
        isinstance(node, ast.Name) and node.id in {"CASES", "case_id"}
        for node in ast.walk(module)
    ):
        raise ValueError(
            f"{definition_name}: v6 forbids selector tables in {label}"
        )
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    if "flag_gems" in imported_roots:
        raise ValueError(
            f"{definition_name}: {label} must be standalone and cannot import "
            "flag_gems"
        )
    return module, top_level_functions(module)


def fixed_float_casts(function: ast.FunctionDef) -> set[str]:
    """Find fixed floating-point conversions in the timing baseline run()."""

    casts: set[str] = set()
    float_names = {"float16", "bfloat16", "float32", "float64"}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(
            node.func, ast.Attribute
        ):
            continue
        if node.func.attr in {"half", "bfloat16", "float", "double"}:
            casts.add(node.func.attr)
        if node.func.attr != "to":
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Attribute)
                and isinstance(argument.value, ast.Name)
                and argument.value.id == "torch"
                and argument.attr in float_names
            ):
                casts.add(f"to(torch.{argument.attr})")
    return casts
