#!/usr/bin/env python3
"""冻结逐元素缓冲候选；只设置当前kernel选项，不修改编译器安装。"""
import ast
import argparse
import hashlib
import json
from pathlib import Path


def prepare(batch, suffixes=None):
    variants = {
        'ubsave-v1': dict(enable_ubuf_saving=True),
        'ubsave-v2': dict(enable_ubuf_saving=True, multibuffer=False,
                          disable_size_align_for_cast=True),
        'mechanism-v3': None,
    }
    records = []
    for op in ['log10_', 'selu_']:
        source = batch/'seeds'/(op+'.py')
        text = source.read_text(encoding='utf-8')
        for suffix, options in variants.items():
            if suffix not in (suffixes or ['ubsave-v1', 'ubsave-v2']):
                continue
            target = batch/'candidates'/(op+'-'+suffix+'.py')
            if target.exists():
                raise FileExistsError(target)
            modified = (text.replace('_BLOCK_MAX = 32768', '_BLOCK_MAX = 65536')
                        if op == 'log10_' else text.replace(
                            'maximum = 16384 if dtype is torch.float16 else 8192',
                            'maximum = 16384'))
            if suffix == 'mechanism-v3':
                if op == 'log10_':
                    options = dict(enable_auto_bind_sub_block=False)
                    modified = text.replace('def _pick_block(n):', 'def _pick_block(n, dtype):')
                    start = modified.index('def _pick_block(')
                    end = modified.index('\ndef _resolve_stream', start)
                    old = modified[start:end]
                    new = old.replace('_BLOCK_MAX', 'maximum')
                    new = new.replace('    if n >=',
                        '    maximum = 32768 if dtype is torch.float16 else 16384\n    if n >=', 1)
                    modified = modified[:start]+new+modified[end:]
                    modified = modified.replace('block = _pick_block(n)', 'block = _pick_block(n, A.dtype)')
                else:
                    options = dict(enable_select_analysis=False, multibuffer=False)
            tree = ast.parse(modified)
            count = 0
            names = {'_'+op.rstrip('_')+'_masked', '_'+op.rstrip('_')+'_nomask'}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                warmup = (isinstance(node.func, ast.Attribute) and node.func.attr == 'warmup'
                          and isinstance(node.func.value, ast.Name) and node.func.value.id in names)
                launch = (isinstance(node.func, ast.Subscript)
                          and isinstance(node.func.value, ast.Name) and node.func.value.id in names)
                if warmup or launch:
                    node.keywords.extend(ast.keyword(arg=k, value=ast.Constant(value=v)) for k, v in options.items())
                    count += 1
            if count != 4:
                raise ValueError('没有匹配全部连续路径入口: '+op)
            ast.fix_missing_locations(tree)
            target.write_text('# Copyright 2026 FlagOS Contributors\n# SPDX-License-Identifier: Apache-2.0\n'
                              '# Codex缓冲机制候选；父版本及编译选项见candidate-provenance.json。\n'
                              + ast.unparse(tree)+'\n', encoding='utf-8')
            records.append(dict(operator=op, suffix=suffix, parent=source.as_posix(),
                parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source=target.as_posix(), source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                options=options, contribution='Codex根据上一轮成对profiling与UB溢出提出kernel选项候选'))
    provenance = batch/('candidate-provenance.json' if suffixes is None else 'candidate-provenance-additional.json')
    if provenance.exists():
        raise FileExistsError(provenance)
    provenance.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--suffixes', nargs='+', choices=['ubsave-v1', 'ubsave-v2', 'mechanism-v3'])
    args = parser.parse_args()
    prepare(args.batch_dir, args.suffixes)
