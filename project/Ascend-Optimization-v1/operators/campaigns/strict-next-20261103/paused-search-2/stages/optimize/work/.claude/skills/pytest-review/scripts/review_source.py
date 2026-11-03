#!/usr/bin/env python3
"""Read-only source triage. Findings are review leads, never semantic approval."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

PROTECTED = ('torch', 'torch_npu', 'torch_musa', 'flag_gems', 'triton', 'pytest')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(path, *, candidate=False):
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = node.module + '.' + item.name

    def name(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return name(node.value) + '.' + node.attr
        if isinstance(node, ast.Subscript):
            return name(node.value)
        return ''

    def protected(value):
        return value.split('.')[0] in PROTECTED

    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def references_gems(node, seen=None):
        seen = set() if seen is None else seen
        for part in ast.walk(node):
            if name(part).startswith('flag_gems.'):
                return True
            if isinstance(part, ast.Name) and part.id in functions and part.id not in seen:
                seen.add(part.id)
                if references_gems(functions[part.id], seen):
                    return True
        return False

    findings = []

    def add(node, code, detail):
        findings.append({'file': str(path), 'line': node.lineno, 'code': code, 'detail': detail})

    for node in ast.walk(tree):
        if candidate:
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, (ast.Attribute, ast.Subscript)) and protected(name(part)):
                        add(node, 'PROTECTED_STATE_WRITE', name(part))
                        break
            if isinstance(node, ast.Call):
                callee = name(node.func)
                if callee in ('setattr', 'delattr') and node.args and protected(name(node.args[0])):
                    add(node, 'PROTECTED_STATE_WRITE', ast.unparse(node))
                elif protected(callee) and any(x in callee for x in ('.register', '.Library', '.impl', '.set_default', '.manual_seed', '.set_rng_state', '.set_grad_enabled')):
                    add(node, 'GLOBAL_STATE_OR_REGISTRATION', callee)
        else:
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == 'torch_op' and references_gems(kw.value):
                        add(node, 'GEMS_BASELINE_REQUIRES_ISOLATION', ast.unparse(kw.value))
                if name(node.func).endswith(('.manual_seed', '.seed')) and any('time' in ast.unparse(a) for a in node.args):
                    add(node, 'TIME_DEPENDENT_SEED', ast.unparse(node))
            if isinstance(node, ast.Attribute) and name(node).startswith('torch.ops.'):
                add(node, 'DIRECT_ATEN_ROUTE_REVIEW', name(node))
    return {'file': str(path), 'sha256': sha256(path), 'findings': findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--test', action='append', type=Path, default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.candidate and not args.test:
        parser.error('provide --candidate or --test')
    try:
        files = ([scan(args.candidate, candidate=True)] if args.candidate else []) + [scan(p) for p in args.test]
        result = {'schema': 'pytest-review-source/v1', 'status': 'REVIEW_REQUIRED', 'files': files,
                  'limits': 'AST triage only; no runtime, numerical correctness or absence-of-side-effects proof.'}
    except (OSError, SyntaxError) as exc:
        result = {'schema': 'pytest-review-source/v1', 'status': 'ERROR', 'error': str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return 2 if result['status'] == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
