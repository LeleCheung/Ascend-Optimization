#!/usr/bin/env python3
"""在CC host版上保留BF16正半轴范围，负半轴用FP16临时量压缩缓冲。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--nan-safe',action='store_true',help='使用where限定负半轴并传播NaN，不依赖minimum的NaN语义')
    a=p.parse_args()
    raw=a.parent.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='9ac565b2a89fc4f4da3c38bfd231cf2da46e57705344d596d51ac27a34c27aa8':
        raise ValueError('CC第二轮父版不同')
    suffix='range-bf16-v7' if a.nan_safe else 'range-bf16-v6'
    target=a.batch_dir/'candidates'/('selu_-'+suffix+'.py')
    if target.exists():raise FileExistsError(target)
    tree=ast.parse(raw.decode())
    branch=ast.parse('''if X.dtype.element_ty == tl.bfloat16:
    positive = (x.to(tl.float32) * 1.0507009873554805).to(tl.bfloat16)
    h = tl.minimum(x.to(tl.float32), 0.).to(tl.float16)
    negative = (1.0507009873554805 * (1.6732632423543772 * (tl.exp(h) - 1.0))).to(tl.bfloat16)
    y = tl.where(x > 0, positive, negative)
''').body[0]
    if a.nan_safe:
        branch=ast.parse(ast.unparse(branch).replace('tl.minimum(x.to(tl.float32), 0.0)',
            'tl.where(x > 0, 0.0, x.to(tl.float32))')).body[0]
    changed=0
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ['_selu_nomask','_selu_masked']:
            cond=next(n for n in node.body if isinstance(n,ast.If))
            replacement=ast.parse(ast.unparse(branch)).body[0]
            replacement.orelse=cond.orelse
            cond.orelse=[replacement]
            changed+=1
    if changed!=2:raise ValueError('连续kernel未全部匹配')
    text=ast.unparse(ast.fix_missing_locations(tree)).replace(
        'maximum = 16384 if dtype is torch.float16 else 8192',
        'maximum = 8192 if dtype is torch.float32 else 16384')
    target.write_text('# Copyright 2026 FlagOS Contributors\n# SPDX-License-Identifier: Apache-2.0\n'
        '# Codex整合：BF16正半轴先FP32缩放并写回BF16；只对非正输入计算半精度指数。\n'+text+'\n',encoding='utf-8')
    record=dict(parent=str(a.parent),parent_sha256=hashlib.sha256(raw).hexdigest(),
        source=str(target),source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        contribution='CC提供host版和BF16半精度方向；Codex重排正负分支并提前收窄输出缓冲，以保留BF16指数范围；必须通过全部门禁')
    record['nan_safe_where']=a.nan_safe
    (a.batch_dir/(suffix+'-provenance.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
