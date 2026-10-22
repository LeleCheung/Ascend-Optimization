#!/usr/bin/env python3
"""为SELU保留同一计算kernel，在采集模式通过带Tensor元数据的launcher执行。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def kernel_ast(text):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body
        if isinstance(n,ast.FunctionDef) and n.name.startswith('_selu_')}


def prepare(parent,output):
    if output.exists():raise FileExistsError(output)
    text=parent.read_text(encoding='utf-8');before=kernel_ast(text)
    old='''        if (getattr(launcher, 'compile_only', False)
                or getattr(launcher, 'enable_msprof_register_tensor', False)
                or getattr(launcher.metadata, 'debug_enabled', False)
                or getattr(launcher.metadata, 'debug', False)):
            return (A.dtype, n, -1, _standard_launch(n, block, masked))
        call = launcher.launch'''
    new='''        requires_tensor = bool(getattr(launcher, 'compile_only', False)
                or getattr(launcher, 'enable_msprof_register_tensor', False)
                or getattr(launcher.metadata, 'debug_enabled', False)
                or getattr(launcher.metadata, 'debug', False))
        # 采集需要本次Tensor元数据，继续执行已选中的同一kernel，不切换persistent策略。
        call = launcher if requires_tensor else launcher.launch'''
    if text.count(old)!=1:raise ValueError('SELU父源码采集分支不同，先核对')
    text=text.replace(old,new)
    old='return (A.dtype, n, kind, call, grid, function, metadata, stream)'
    if text.count(old)!=1:raise ValueError('SELU缓存结构不同')
    text=text.replace(old,'return (A.dtype, n, kind, call, grid, function, metadata, stream, requires_tensor)')
    for indent in ['    ','        ']:
        old=indent+'kind = entry[2]\n'+indent+'if kind == 0:'
        new=indent+'kind = entry[2]\n'+indent+'argument = A if len(entry) > 8 and entry[8] else ptr\n'+indent+'if kind == 0:'
        if text.count(old)!=1:raise ValueError('SELU分派结构不同')
        text=text.replace(old,new)
    count=text.count('None, None, None, ptr,')
    if count not in [4,6]:raise ValueError('SELU裸启动数量不同')
    text=text.replace('None, None, None, ptr,','None, None, None, argument,')
    if kernel_ast(text)!=before:raise ValueError('计算kernel AST发生变化')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(text.encode('utf-8'))
    proof=dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        source=str(output),source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        kernel_ast_unchanged=True,kernels=list(before),launch_sites=count,
        contribution='CC/DeepSeek生成计算与启动方案；Codex修复采集模式同kernel/Tensor元数据传递，正常裸启动保持不变')
    output.with_suffix('.provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.parent,a.output),ensure_ascii=False))
