"""Maintainer-owned, narrow no-JIT exceptions for identity/view operators.

This grants only a source-policy exception. ABI, correctness, alias/effect and
workload validation still apply. The operator comes from the bound definition,
never the candidate's display name or a candidate-supplied policy declaration.
"""
from __future__ import annotations

import ast
from types import MappingProxyType

from ..protocol.schema import Implementation


METADATA_OPERATORS = MappingProxyType({
    "lift_fresh": "identity: return the original input object without mutation",
    "unsqueeze_": "inplace view: insert dim while retaining input object/storage",
})


def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def permits_no_jit(implementation: Implementation, operator_name: str | None) -> bool:
    """Recognize only the reviewed minimal recipes; an operator name is not enough."""
    if operator_name not in METADATA_OPERATORS or len(implementation.sources) != 1:
        return False
    source = implementation.sources[0]
    path, separator, entrypoint = implementation.entrypoint.partition("::")
    if not separator or path != source.path or not source.path.endswith(".py"):
        return False
    try:
        tree = ast.parse(source.content, filename=source.path)
    except SyntaxError:
        return False
    functions = []
    for node in _without_docstring(tree.body):
        if isinstance(node, ast.Import) and all(item.name == "torch" for item in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__" and all(item.name == "annotations" for item in node.names):
            continue
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
        else:
            return False
    if len(functions) != 1:
        return False
    function = functions[0]
    if function.name != entrypoint or function.decorator_list:
        return False
    args = function.args
    positional = args.posonlyargs + args.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
        return False
    # Annotation calls could mutate state at import time. Simple type hints are OK.
    annotations = [arg.annotation for arg in positional] + [function.returns]
    if any(isinstance(node, (ast.Call, ast.NamedExpr)) for annotation in annotations if annotation for node in ast.walk(annotation)):
        return False
    body = _without_docstring(function.body)
    expected_arity = 1 if operator_name == "lift_fresh" else 2
    if len(positional) != expected_arity:
        return False
    tensor_name = positional[0].arg

    def returns_input(node):
        return isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == tensor_name

    if operator_name == "lift_fresh":
        return len(body) == 1 and returns_input(body[0])

    def unsqueeze_call(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == tensor_name
            and node.func.attr == "unsqueeze_"
            and not node.keywords
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == positional[1].arg
        )

    return (
        len(body) == 1 and isinstance(body[0], ast.Return) and unsqueeze_call(body[0].value)
    ) or (
        len(body) == 2 and isinstance(body[0], ast.Expr) and unsqueeze_call(body[0].value)
        and returns_input(body[1])
    )
