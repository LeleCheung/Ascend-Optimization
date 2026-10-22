#!/usr/bin/env python3
"""复用已验证的轻量启动，并应用CC实测的fp16分块；不缓存输入对象或地址。"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def build(source, metadata_transform):
    """从固定log10启动框架构建SELU；未知源码明确拒绝。"""
    tree=ast.parse(source)
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant):
        tree.body[0].value=ast.Constant(value='SELU原地计算：轻量启动、完整stride、按dtype控制分块；本次输入地址和stream每次读取。')
    kernels=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_log10_')]
    if {n.name for n in kernels}!={'_log10_nomask','_log10_masked','_log10_strided'}:
        raise ValueError('父源码kernel结构发生变化')
    changed=[]
    for kernel in kernels:
        for i,node in enumerate(kernel.body):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='y' for t in node.targets):
                calculation='''
if X.dtype.element_ty == tl.float16:
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
else:
    xf = x.to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
'''
                kernel.body[i:i+1]=ast.parse(calculation).body
                changed.append(kernel.name);break
    if len(changed)!=3:
        raise ValueError('未找到全部数值表达式')
    text=ast.unparse(tree).replace('_log10_','_selu_')
    old=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick_block'))
    new='''def _pick_block(n, dtype):
    maximum = 16384 if dtype is torch.float16 else 8192
    if n >= maximum * _BLOCK_TARGET_PROGRAMS:
        return maximum
    target = -(-n // _BLOCK_TARGET_PROGRAMS)
    block = _BLOCK_MIN
    while block < target:
        block <<= 1
    return min(block, maximum)'''
    text=text.replace(old,new).replace('block = _pick_block(n)','block = _pick_block(n, A.dtype)')
    if 'def _pick_block(n):' in text or '_log10_' in text:
        raise ValueError('转换未完成')
    transformed,closures=metadata_transform(text+'\n')
    return ('# Copyright 2026 FlagOS Contributors\n'
            '# SPDX-License-Identifier: Apache-2.0\n'
            '# HG-KDA/CC诊断＋Codex整合：当前地址、当前stream、完整stride；按dtype控制UB分块。\n'
            +transformed),dict(kernels=changed,closures=closures)


def main():
    root=Path('/data/hanle/ascend-optimization')
    batch=root/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/paired-pointwise-20261022'
    state=json.loads((batch/'kg-paired-state.json').read_text())
    if state['status']!='completed':
        raise RuntimeError('先等待现有KG终态')
    parent=batch/'results/log10_/paired-HG-KDA/export/main.py'
    target=batch/'candidates/selu_-profilemeta-v4.py'
    if target.exists():raise ValueError('候选已存在')
    s=importlib.util.spec_from_file_location('paired_meta',Path(__file__).with_name('prepare-profilemeta.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    content,changes=build(parent.read_text(),m.transform)
    target.write_text(content,encoding='utf-8')
    record=dict(operator='selu_',source=str(target),source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        launcher_parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),changes=changes,
        method='复用log10的无输入缓存启动；fp16使用CC实测的native计算和16384块，bf16/fp32保留8192；整数地址提供size元数据',
        attribution='CC分块诊断与log10生成框架＋Codex安全整合；未评测前不算有效成果')
    (batch/'selu-v4-provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
