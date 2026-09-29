#!/usr/bin/env python3
"""Mechanically align kernel_todo ``run`` signatures with a V6 Catalog.

The optimized function body is preserved.  Renamed parameters receive a small
alias prelude so existing kernel code keeps using its local names.  The two
genuinely structural public ABIs in the current corpus (variadic
``broadcast_tensors`` and list-based ``rnn_relu`` parameters) have explicit
adapters below instead of positional guessing.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import tokenize
from pathlib import Path
from typing import Any


def _definitions(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        name = value["name"]
        if name in result:
            raise ValueError(f"duplicate Definition.name: {name}")
        result[name] = value
    return result


def _run_node(source: str) -> ast.FunctionDef:
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    ]
    if len(functions) != 1:
        raise ValueError(f"expected exactly one top-level run(), got {len(functions)}")
    return functions[0]


def _actual_names(function: ast.FunctionDef) -> list[str]:
    names = [
        argument.arg
        for argument in [*function.args.posonlyargs, *function.args.args]
    ]
    if function.args.vararg is not None:
        names.append(function.args.vararg.arg)
    names.extend(argument.arg for argument in function.args.kwonlyargs)
    if function.args.kwarg is not None:
        names.append(function.args.kwarg.arg)
    return names


def _signature_rows(function: ast.FunctionDef) -> list[tuple[str, str, bool, Any]]:
    positional = [*function.args.posonlyargs, *function.args.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(function.args.defaults)
    ) + list(function.args.defaults)
    rows = [
        (
            argument.arg,
            (
                "positional_only"
                if index < len(function.args.posonlyargs)
                else "positional_or_keyword"
            ),
            default is None,
            None if default is None else ast.literal_eval(default),
        )
        for index, (argument, default) in enumerate(zip(positional, defaults))
    ]
    if function.args.vararg is not None:
        rows.append((function.args.vararg.arg, "var_positional", True, None))
    rows.extend(
        (
            argument.arg,
            "keyword_only",
            default is None,
            None if default is None else ast.literal_eval(default),
        )
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults
        )
    )
    if function.args.kwarg is not None:
        rows.append((function.args.kwarg.arg, "var_keyword", True, None))
    return rows


def _definition_rows(definition: dict[str, Any]) -> list[tuple[str, str, bool, Any]]:
    return [
        (
            parameter["name"],
            parameter.get("kind", "positional_or_keyword"),
            parameter["required"],
            parameter.get("default"),
        )
        for parameter in definition["parameters"]
    ]


def _format_parameter(parameter: dict[str, Any]) -> str:
    name = parameter["name"]
    if parameter.get("kind") == "var_positional":
        return f"*{name}"
    if parameter["required"]:
        return name
    return f"{name}={parameter['default']!r}"


def _format_signature(parameters: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    inserted_star = False
    for parameter in parameters:
        kind = parameter.get("kind", "positional_or_keyword")
        if kind == "keyword_only" and not inserted_star:
            pieces.append("*")
            inserted_star = True
        pieces.append(_format_parameter(parameter))
        if kind == "var_positional":
            inserted_star = True
    compact = f"def run({', '.join(pieces)}):"
    if len(compact) <= 88:
        return compact
    return "def run(\n" + "".join(f"    {piece},\n" for piece in pieces) + "):"


def _offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _absolute(offsets: list[int], position: tuple[int, int]) -> int:
    row, column = position
    return offsets[row - 1] + column


def _header_span(source: str, function: ast.FunctionDef) -> tuple[int, int]:
    offsets = _offsets(source)
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    found_def = False
    found_run = False
    depth = 0
    start = None
    for token in tokens:
        if token.start[0] < function.lineno:
            continue
        if not found_def:
            if token.type == tokenize.NAME and token.string == "def":
                found_def = True
                start = _absolute(offsets, token.start)
            continue
        if not found_run:
            if token.type == tokenize.NAME and token.string == "run":
                found_run = True
            continue
        if token.string in "([{":
            depth += 1
        elif token.string in ")]}":
            depth -= 1
        elif token.string == ":" and depth == 0:
            assert start is not None
            return start, _absolute(offsets, token.end)
    raise ValueError("could not locate run() header")


def _insertion_offset(source: str, function: ast.FunctionDef) -> int:
    body = function.body
    index = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        index = 1
    node = body[index] if index < len(body) else body[-1]
    return _absolute(_offsets(source), (node.lineno, node.col_offset))


def _alias_lines(
    actual_names: list[str], expected_names: list[str]
) -> list[str]:
    if len(actual_names) > len(expected_names):
        raise ValueError("structural ABI needs an explicit adapter")
    exact = set(actual_names).intersection(expected_names)
    available = [name for name in expected_names if name not in exact]
    aliases: list[str] = []
    for old in actual_names:
        if old in exact:
            continue
        if not available:
            raise ValueError(f"cannot map old parameter {old!r}")
        new = available.pop(0)
        aliases.append(f"{old} = {new}")
    return aliases


def _replace_signature(
    source: str,
    function: ast.FunctionDef,
    definition: dict[str, Any],
    aliases: list[str],
) -> str:
    header_start, header_end = _header_span(source, function)
    insertion = _insertion_offset(source, function)
    if aliases:
        indent = " " * function.body[0].col_offset
        prelude = (
            "# KernelGen V6 ABI compatibility aliases.\n"
            + indent
            + ("\n" + indent).join(aliases)
            + "\n"
            + indent
        )
        source = source[:insertion] + prelude + source[insertion:]
    signature = _format_signature(definition["parameters"])
    return source[:header_start] + signature + source[header_end:]


def _migrate_broadcast(source: str, function: ast.FunctionDef) -> str:
    offsets = _offsets(source)
    name_start = _absolute(offsets, (function.lineno, function.col_offset + 4))
    name_end = name_start + len("run")
    source = source[:name_start] + "_run_pair" + source[name_end:]
    wrapper = '''

def run(*tensors):
    if not tensors:
        raise ValueError("broadcast_tensors expects at least one tensor")
    expanded = [tensors[0]]
    for tensor in tensors[1:]:
        next_expanded = []
        expanded_tensor = tensor
        for previous in expanded:
            previous, expanded_tensor = _run_pair(previous, tensor)
            next_expanded.append(previous)
        next_expanded.append(expanded_tensor)
        expanded = next_expanded
    return tuple(expanded)
'''
    return source.rstrip() + wrapper


def _migrate_one(path: Path, definition: dict[str, Any]) -> bool:
    source = path.read_text(encoding="utf-8")
    function = _run_node(source)
    if _signature_rows(function) == _definition_rows(definition):
        return False
    operator = definition["name"]
    actual_names = _actual_names(function)
    expected_names = [item["name"] for item in definition["parameters"]]
    if operator == "broadcast_tensors":
        migrated = _migrate_broadcast(source, function)
    elif operator == "rnn_relu":
        migrated = _replace_signature(
            source,
            function,
            definition,
            [
                "if params is None or len(params) < 2:",
                "    raise ValueError('rnn_relu params must contain input/hidden weights')",
                "w_ih = params[0]",
                "w_hh = params[1]",
                "b_ih = params[2] if has_biases else None",
                "b_hh = params[3] if has_biases else None",
            ],
        )
    else:
        migrated = _replace_signature(
            source,
            function,
            definition,
            _alias_lines(actual_names, expected_names),
        )
    path.write_text(migrated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-todo", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, required=True)
    args = parser.parse_args()
    definitions = _definitions(args.definitions.resolve())
    changed = 0
    missing: list[str] = []
    for operator, definition in definitions.items():
        paths = sorted(args.kernel_todo.resolve().glob(f"*/codes/{operator}.py"))
        if not paths:
            missing.append(operator)
        for path in paths:
            changed += _migrate_one(path, definition)
    print(
        f"definitions={len(definitions)} changed_files={changed} "
        f"missing_operators={len(missing)}"
    )
    if missing:
        print("missing: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
