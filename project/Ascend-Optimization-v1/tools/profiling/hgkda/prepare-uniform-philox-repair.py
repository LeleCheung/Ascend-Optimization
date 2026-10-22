#!/usr/bin/env python3
"""在独立副本修复已证实的Philox常数、低字拼接和counter进位；不回写CC源码。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(parent, output):
    if output.exists():
        raise FileExistsError(output)
    text = parent.read_text(encoding='utf-8')
    before = ast.parse(text)
    functions = {node.name:ast.dump(node,include_attributes=False) for node in before.body if isinstance(node,ast.FunctionDef)}
    if not {'_mul_hi_lo','_philox4','uniform_kernel'} <= set(functions):
        raise ValueError('源码结构已变化，不能盲目替换')
    changes = []
    old = 'w0 = (tl.full((), 0x9E37, tl.uint32) << 16) + tl.full((), 0xB9, tl.uint32)'
    new = 'w0 = (tl.full((), 0x9E37, tl.uint32) << 16) + tl.full((), 0x79B9, tl.uint32)'
    if text.count(old)==1:
        text=text.replace(old,new)
        changes.append('Weyl常数修正为0x9E3779B9')
    elif new not in text:
        raise ValueError('Weyl常数表达式不同，需独立审查')
    old = '    lo = u0 ^ (w1 << 16)'
    if text.count(old)==1:
        text=text.replace(old,'    lo = (u0 & 0xFFFF) | ((w1 & 0xFFFF) << 16)')
        changes.append('低32位按半字拼接，消除重复异或')
    elif 'lo = (u0 & 0xFFFF) | ((w1 & 0xFFFF) << 16)' not in text:
        raise ValueError('低字表达式不同，需独立审查')
    old = '    c0 = c0 + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)'
    new = '''    lane = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.uint32)
    new_c0 = c0 + lane
    c1 = c1 + (new_c0 < c0).to(tl.uint32)
    c0 = new_c0'''
    if text.count(old)==1:
        text=text.replace(old,new)
        changes.append('counter低32位溢出时向高32位进位')
    elif new not in text:
        raise ValueError('counter表达式不同，需独立审查')
    if not changes:
        raise ValueError('没有发现待修复差异，不复制相同源码进行重复评测')
    after = ast.parse(text)
    functions_after = {node.name:ast.dump(node,include_attributes=False) for node in after.body if isinstance(node,ast.FunctionDef)}
    changed = {name for name in functions if functions[name]!=functions_after.get(name)}
    if set(functions)!=set(functions_after) or not changed <= {'_mul_hi_lo','_philox4','uniform_kernel'}:
        raise ValueError('修复超出登记的计算函数')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(text.encode('utf-8'))
    data = dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                source=str(output),source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                changes=changes,changed_functions=sorted(changed),
                contribution='CC/DeepSeek生成向量化随机数、边界与启动方案；Codex独立修复Philox整数机制。须通过独立逐位NPU门禁和原有全部语义及一次KGS后方可交付。')
    output.with_suffix('.provenance.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.parent,a.output),ensure_ascii=False))
