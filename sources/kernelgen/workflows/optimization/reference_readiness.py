"""Build a reference candidate without copying evaluator-owned hooks."""

import ast

from kernelgen_client import Implementation, SourceFile


def reference_implementation(operator):
    tree = ast.parse(operator.definition.reference)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    primary = "correctness_run" if "correctness_run" in functions else "run"
    if primary not in functions:
        raise ValueError("reference readiness requires correctness_run or run")
    hooks = {"gen_inputs", "valid"} & functions
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name in hooks)]
    # A callable that depends on a removed hook cannot be copied faithfully.
    # Fail explicitly instead of changing its behavior or bypassing admission.
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} & hooks
    if referenced:
        raise ValueError("reference callable depends on evaluator hooks: " + ", ".join(sorted(referenced)))
    source = ast.unparse(tree) + "\n"
    sources = [SourceFile(path="main.py", content=source)]
    if primary != "run":
        # Protocol entrypoints are named run. A separate module preserves any
        # original run used by correctness_run instead of rebinding its globals.
        sources = [SourceFile(path="main.py", content=f"from reference_impl import {primary} as run\n"),
                   SourceFile(path="reference_impl.py", content=source)]
    return Implementation(
        name="catalog-reference-readiness", definition=operator.definition.name,
        language="python", entrypoint="main.py::run", sources=sources,
    )
