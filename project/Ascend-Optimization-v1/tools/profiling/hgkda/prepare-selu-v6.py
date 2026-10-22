#!/usr/bin/env python3
"""用固定编译器的kernel选项检验多缓冲是否阻止BF16/FP32扩大分块。"""
import ast
import hashlib
import json
from pathlib import Path


def main():
    b=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/paired-pointwise-20261022')
    parent=b/'candidates/selu_-profilemeta-v4.py';target=b/'candidates/selu_-profilemeta-v6.py'
    if target.exists():raise ValueError('候选已存在')
    text=parent.read_text()
    old='maximum = 16384 if dtype is torch.float16 else 8192'
    if text.count(old)!=1:raise ValueError('父源码分块结构不同')
    tree=ast.parse(text.replace(old,'maximum = 16384'))
    changed=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        warmup=(isinstance(node.func,ast.Attribute) and node.func.attr=='warmup'
                and isinstance(node.func.value,ast.Name) and node.func.value.id in {'_selu_nomask','_selu_masked'})
        standard=(isinstance(node.func,ast.Subscript) and isinstance(node.func.value,ast.Name)
                  and node.func.value.id in {'_selu_nomask','_selu_masked'})
        if warmup or standard:
            node.keywords.append(ast.keyword(arg='multibuffer',value=ast.Constant(value=False)))
            changed.append(ast.unparse(node.func))
    if len(changed)!=4:raise ValueError('未找到全部启动/回退入口')
    ast.fix_missing_locations(tree)
    header='# Copyright 2026 FlagOS Contributors\n# SPDX-License-Identifier: Apache-2.0\n# 仅本kernel关闭自动多缓冲，验证UB与分块机制；未改变安装的编译器。\n'
    target.write_text(header+ast.unparse(tree)+'\n',encoding='utf-8')
    v=dict(operator='selu_',parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        source=str(target),source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changed=changed,
        method='全部dtype尝试16384块，dense kernel设置multibuffer=False；完整语义/编译失败即不交付',
        attribution='Codex根据CC的UB失败与IR资源记录追加的编译机制实验；固定FlagTree安装不变')
    (b/'selu-v6-provenance.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(v,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
