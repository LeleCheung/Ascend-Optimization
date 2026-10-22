#!/usr/bin/env python3
"""组合CC实测有效的四块加载与父版当前地址启动，FP32保留父版路径。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


HELPER = '''
def _make_multi_launch(A, n):
    grid = n // 32768
    def standard(X, ptr):
        _log10_x4[(grid,)](X, BLOCK=8192)
    if A.data_ptr() & 15:
        return standard
    try:
        compiled = _log10_x4.warmup(A, BLOCK=8192, grid=(1,))
        launcher = compiled.run
        if (getattr(launcher, 'compile_only', False)
            or getattr(launcher, 'enable_msprof_register_tensor', False)
            or getattr(launcher.metadata, 'debug_enabled', False)):
            return standard
        call, function, metadata = launcher.launch, compiled.function, compiled.packed_metadata
        stream = _resolve_stream(A.device.index)
    except Exception:
        return standard
    def launch(X, ptr, _address=_hgkda_address_type(n)):
        call(grid, 1, 1, stream(), function, metadata, None, None, None, _address(ptr), None)
    return launch
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir', type=Path, required=True)
    p.add_argument('--generated-source', type=Path, required=True)
    p.add_argument('--cached-host', type=Path, help='使用CC第二轮完整地址缓存启动，而非父版按形状启动')
    a = p.parse_args()
    suffix='merged-v8' if a.cached_host else 'pipeline-v4'
    target = a.batch_dir/'candidates'/('log10_-'+suffix+'.py')
    if target.exists():
        raise FileExistsError(target)
    raw = a.generated_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '8cbbaf3edb4978a8dc6abc46c98ebcfef02bb2dc3029f4ef257c789be35dee13':
        raise ValueError('CC第三轮字节已变化，需重新审查')
    kernels = [n for n in ast.parse(raw.decode()).body if isinstance(n, ast.FunctionDef) and n.name == '_log10_x4']
    if len(kernels) != 1:
        raise ValueError('四块kernel不唯一')
    seed = a.batch_dir/'seeds/log10_.py'
    if a.cached_host:
        seed=a.cached_host
        if hashlib.sha256(seed.read_bytes()).hexdigest() != '2b208ae4c2c4394549f03fc466768c1f280011516f34878ab49998d23825b6fd':
            raise ValueError('CC第二轮host源码不同')
    text = seed.read_text(encoding='utf-8')
    text = text.replace('@triton.jit\ndef _log10_masked', ast.unparse(kernels[0])+'\n\n@triton.jit\ndef _log10_masked', 1)
    anchor = 'def _make_launch(A, n, ptr):' if a.cached_host else 'def _make_launch(A, n):'
    if text.count(anchor) != 1:
        raise ValueError('父版启动结构不同')
    helper=HELPER
    if a.cached_host:
        helper=helper.replace('def _make_multi_launch(A, n):','def _make_multi_launch(A, n, ptr):')
        helper=helper.replace('    def launch(X, ptr, _address=_hgkda_address_type(n)):',
            '    address = _hgkda_address_type(n)(ptr)\n    def launch(X, ptr):')
        helper=helper.replace('_address(ptr), None)', 'address, None)')
    text = text.replace(anchor, helper+'\n'+anchor, 1)
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_make_launch':
            gate = ast.parse('''if (A.dtype is torch.float16 or A.dtype is torch.bfloat16) and n >= 4194304 and n % 32768 == 0:
    return _make_multi_launch(A, n)
''').body[0]
            if a.cached_host:
                gate=ast.parse(ast.unparse(gate).replace('_make_multi_launch(A, n)',
                    '_make_multi_launch(A, n, ptr)')).body[0]
            node.body.insert(1 if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else 0, gate)
    tree.body[0] = ast.Expr(value=ast.Constant(value='log10_：大FP16/BF16采用CC四块加载，FP32及其余输入沿用父版；每次读取当前地址和stream。'))
    ast.fix_missing_locations(tree)
    target.write_text('# Copyright 2026 FlagOS Contributors\n# SPDX-License-Identifier: Apache-2.0\n'+ast.unparse(tree)+'\n', encoding='utf-8')
    record = dict(source=str(target), source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        parent_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(), cc_round3_sha256=hashlib.sha256(raw).hexdigest(),
        contribution='CC生成并实测四块加载；Codex保留父版当前地址启动、限制到FP16/BF16大输入，并使profiling回退使用同一kernel')
    record['cached_host']=bool(a.cached_host)
    if a.cached_host:
        record['contribution']='CC提供完整地址键的启动缓存与四块加载；Codex将其组合并保留FP32单块路径、使profiling调用同一kernel'
    (a.batch_dir/(suffix+'-provenance.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
