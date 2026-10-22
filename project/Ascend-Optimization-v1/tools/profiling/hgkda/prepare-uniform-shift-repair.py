#!/usr/bin/env python3
"""绕过固定后端无符号右移的已测反例，显式使用有符号移位与低位掩码。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(parent,output):
    if output.exists():raise FileExistsError(output)
    text=parent.read_text(encoding='utf-8')
    tree=ast.parse(text)
    functions={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    if '_mul_hi_lo' not in functions:raise ValueError('没有已审查的乘法拆分函数')
    for name in ['a','u0','t','w1']:
        old=name+' >> 16'
        # 此形式的掩码对signed shift在语义上必需，不能当作unsigned shift冗余掩码消除。
        new='(('+name+'.to(tl.int32, bitcast=True) >> 16) & 0xFFFF).to(tl.uint32)'
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mul_hi_lo')
        lines=text.splitlines(True)
        following=next((n.lineno-1 for n in tree.body if n.lineno>node.lineno),len(lines))
        begin=node.lineno-1
        function=''.join(lines[begin:following])
        if function.count(old)!=1:raise ValueError('移位结构不同: '+name)
        function=function.replace(old,new)
        text=''.join(lines[:begin])+function+''.join(lines[following:])
        tree=ast.parse(text)
    after={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    changed={name for name in functions if functions[name]!=after.get(name)}
    if set(functions)!=set(after) or changed!={'_mul_hi_lo'}:raise ValueError('修改超出已登记整数拆分函数')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(text.encode('utf-8'))
    proof=dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               source=str(output),source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
               changed_functions=sorted(changed),contribution='Codex根据64组设备整数原语反例，显式signed shift+mask绕过逻辑右移错误；CC原分块与启动方案保持。尚须逐位设备核验、原门禁与一次完整KGS。')
    output.with_suffix('.provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.parent,a.output),ensure_ascii=False))
