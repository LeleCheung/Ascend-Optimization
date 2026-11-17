"""Classify displayed version differences from frozen source ASTs.

This describes changed code, not the evidence that motivated the change.
Imports, comments and docstrings do not count as executable changes.
"""
import ast
import hashlib
import json
from pathlib import Path


def sections(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    device = {name for name, node in functions.items()
              if any('jit' in ast.unparse(d) for d in node.decorator_list)}
    while True:
        calls = {n.func.id for name in device for n in ast.walk(functions[name])
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        expanded = device | (calls & functions.keys())
        if expanded == device:
            break
        device = expanded
    result = {'kernel': {}, 'host': {}, 'shared': {}}
    device_names = {n.id for name in device for statement in functions[name].body
                    for n in ast.walk(statement) if isinstance(n, ast.Name)}
    host_names = {n.id for name in functions.keys() - device for n in ast.walk(functions[name])
                  if isinstance(n, ast.Name)}
    for node in ast.walk(tree):
        if hasattr(node, 'body') and isinstance(node.body, list):
            node.body = [n for n in node.body if not (isinstance(n, ast.Expr)
                         and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            category = 'kernel' if node.name in device else 'host'
            key = node.name
            if category == 'kernel':
                decorators = [d for d in node.decorator_list if 'jit' not in ast.unparse(d)]
                if decorators:
                    result['host'][key + ':launch_decorators'] = '|'.join(ast.dump(d) for d in decorators)
                node.decorator_list = []
        else:
            key = ast.unparse(node.targets[0]) if isinstance(node, ast.Assign) else str(i)
            # Module caches, imports and launch setup execute on the host.
            # Constants referenced inside kernels affect device code as well.
            category = ('shared' if key in device_names and key in host_names
                        else 'kernel' if key in device_names else 'host')
        result[category][key] = ast.dump(node, include_attributes=False)
    return result


def annotate(rows, root):
    index = json.loads((root / 'development/experiments/lineage/20261117/node-source-index.json').read_bytes())
    cache = {}
    def source(sha):
        if sha not in cache:
            path = root / index[sha]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
            cache[sha] = sections(path)
        return cache[sha]
    for row in rows:
        previous = None
        for event in row['display_events']:
            sha = event['source_sha256']
            current = source(sha)
            detail = {'source_sha256': sha, 'source_path': index[sha],
                      'basis': 'AST difference from preceding displayed version; not causal attribution'}
            if previous is None:
                location = 'baseline'
            else:
                parent_sha = previous['source_sha256']
                parent = source(parent_sha)
                changes = {part: sorted(k for k in current[part].keys() | parent[part].keys()
                                       if current[part].get(k) != parent[part].get(k))
                           for part in current}
                detail.update(parent_sha256=parent_sha, changed_symbols=changes)
                kernel, host, shared = (bool(changes[k]) for k in ('kernel', 'host', 'shared'))
                location = ('mixed' if shared or (kernel and host) else 'kernel' if kernel
                            else 'host' if host else 'retest')
            detail['location'] = location
            event['change_location'] = detail
            previous = event
