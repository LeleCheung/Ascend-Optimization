#!/usr/bin/env python3
"""验证低精度log计算能否降低大输入设备时间；保持通用启动、stride和原合同。"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    root=Path('/data/hanle/ascend-optimization')
    batch=root/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/paired-pointwise-20261022'
    parent=batch/'results/log10_/paired-HG-KDA/export/main.py'
    target=batch/'candidates/log10_-profilemeta-v5.py'
    if target.exists():raise ValueError('候选已存在')
    tree=ast.parse(parent.read_text())
    changed=[]
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef) or not node.name.startswith('_log10_'):continue
        for item in ast.walk(node):
            if isinstance(item,ast.Call) and isinstance(item.func,ast.Attribute) and item.func.attr=='log':
                if (len(item.args)!=1 or not isinstance(item.args[0],ast.Call)
                    or not isinstance(item.args[0].func,ast.Attribute) or item.args[0].func.attr!='to'):
                    raise ValueError('父源码log表达式不符合已知结构')
                item.args[0]=ast.Name(id='x',ctx=ast.Load());changed.append(node.name)
    if len(changed)!=3:raise ValueError('未找到全部log数学表达式')
    s=importlib.util.spec_from_file_location('log10_v5_meta',Path(__file__).with_name('prepare-profilemeta.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    ast.fix_missing_locations(tree)
    source,closures=m.transform(ast.unparse(tree)+'\n')
    header=parent.read_text().split('"""',1)[0]
    target.write_text(header+source,encoding='utf-8')
    record=dict(operator='log10_',parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        source=str(target),source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changed=changed,closures=closures,
        method='保持32768分块和本次地址启动，移除log前无条件FP32转换，检验低精度数学及缓冲成本',
        attribution='Codex根据成对硬件计数和IR追加的单项数值实现实验；必须通过42＋12及原36项才能交付')
    (batch/'log10-v5-provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
