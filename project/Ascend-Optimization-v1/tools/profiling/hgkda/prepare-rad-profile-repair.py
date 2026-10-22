#!/usr/bin/env python3
"""在独立副本修复CC rad2deg裸启动的采集元数据和N整除缓存键，不改计算kernel。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(parent,output):
    if output.exists():raise FileExistsError(output)
    text=parent.read_text(encoding='utf-8')
    original=ast.parse(text)
    kernels={n.name:ast.dump(n,include_attributes=False) for n in original.body
        if isinstance(n,ast.FunctionDef) and n.name.startswith('_rad2deg_')}
    old='    return (compiled.run.launch, compiled.function, compiled.packed_metadata, block)'
    new='''    launcher = compiled.run
    if (getattr(launcher, 'enable_msprof_register_tensor', False)
            or getattr(launcher, 'compile_only', False)
            or getattr(getattr(launcher, 'metadata', None), 'debug_enabled', False)):
        # 本机采集launcher需要Tensor元数据；回到相同kernel的标准启动，不切换计算。
        return (None, None, None, block)
    return (launcher.launch, compiled.function, compiled.packed_metadata, block)'''
    if text.count(old)!=1:raise ValueError('父源码绑定入口不同，先审查')
    text=text.replace(old,new)
    old='''        entry[0]((n + block - 1) // block, 1, 1, _stream_fn(),
                 entry[1], entry[2], None, None, None, ptr, n, block)'''
    new='''        if entry[0] is None:
            _rad2deg_contig_kernel[((n + block - 1) // block,)](A, n, BLOCK=block)
        else:
            entry[0]((n + block - 1) // block, 1, 1, _stream_fn(),
                     entry[1], entry[2], None, None, None, ptr, n, block)'''
    if text.count(old)!=1:raise ValueError('父源码直接启动入口不同，先审查')
    text=text.replace(old,new)
    if text.count('(A.dtype, ptr & 15, block)')!=2:raise ValueError('父源码缓存键不同')
    text=text.replace('(A.dtype, ptr & 15, block)','(A.dtype, ptr & 15, block, n & 15)')
    result=ast.parse(text)
    after={n.name:ast.dump(n,include_attributes=False) for n in result.body
        if isinstance(n,ast.FunctionDef) and n.name.startswith('_rad2deg_')}
    if kernels!=after:raise ValueError('计算kernel AST发生变化')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(text.encode('utf-8'))
    proof=dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        source=str(output),source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        kernels=list(kernels),kernel_ast_unchanged=True,
        contribution='CC/DeepSeek生成计算与启动方案；Codex追加采集元数据兼容及N模16缓存身份修复，不更改算法或计算kernel')
    output.with_suffix('.provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.parent,a.output),ensure_ascii=False))
