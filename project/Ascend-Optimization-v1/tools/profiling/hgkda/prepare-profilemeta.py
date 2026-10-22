#!/usr/bin/env python3
"""保留真实地址的整数入口，同时给后端profiler提供size元数据。"""
import ast
import hashlib
import json
from pathlib import Path
import sys


FACTORY = '''
def _hgkda_address_type(n):
    class DeviceAddress(int):
        def size(self):
            return (n,)
    return DeviceAddress
'''


def transform(source):
    tree = ast.parse(source)
    changed = []

    class Visitor(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'call'
                     and any(isinstance(arg, ast.Name) and arg.id == 'ptr' for arg in n.args)]
            if node.name not in {'launch', 'invoke'} or not calls:
                return node
            # 类型只按编译配置创建一次；每次调用用本次ptr构造整数实例，不保留输入。
            node.args.kwonlyargs.append(ast.arg(arg='_address'))
            node.args.kw_defaults.append(ast.Call(func=ast.Name(id='_hgkda_address_type', ctx=ast.Load()),
                                                   args=[ast.Name(id='n', ctx=ast.Load())], keywords=[]))
            for call in calls:
                call.args = [ast.Call(func=ast.Name(id='_address', ctx=ast.Load()), args=[arg], keywords=[])
                             if isinstance(arg, ast.Name) and arg.id == 'ptr' else arg for arg in call.args]
            changed.append(node.name)
            return node

    tree = Visitor().visit(tree)
    if not changed:
        raise ValueError('没有找到已核实的原始地址launcher，拒绝猜测替换')
    # 工厂在定义闭包前已经可用；不修改数值计算IR或计分逻辑。
    imports_end = 1 if tree.body and isinstance(tree.body[0], ast.Expr) else 0
    tree.body[imports_end:imports_end] = ast.parse(FACTORY).body
    ast.fix_missing_locations(tree)
    header = source.split('"""', 1)[0] if source.startswith('# Copyright') else ''
    return header+ast.unparse(tree)+'\n', changed


def main():
    if sys.version_info < (3, 9):
        raise SystemExit('此转换需要ast.unparse；请在tle_yy容器使用Python 3.11，不使用宿主Python 3.7。')
    root = Path('/data/hanle/ascend-optimization')
    batch = root/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/paired-pointwise-20261022'
    import fcntl
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((batch/'kg-paired-state.json').read_text())
        if state['status'] != 'completed':
            raise RuntimeError('原KG队列尚未完整结束')
        records = []
        for op in ['log10_', 'selu_']:
            arm = state['results'].get(op, {})
            source = (batch/'results'/op/'paired-HG-KDA/export/main.py' if arm.get('status') == 'passed'
                      else batch/'candidates'/(op+'-rawptr-v1.py'))
            target = batch/'candidates'/(op+'-profilemeta-v3.py')
            if target.exists():
                raise ValueError('追加候选已存在')
            content, changed = transform(source.read_text())
            target.write_text(content, encoding='utf-8')
            records.append(dict(operator=op, parent=str(source),
                parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source=str(target),
                source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), changed_closures=changed,
                mechanism='PyLong入口不变；整数子类提供size()，适配固定后端profiler元数据读取；尚需实测'))
        path = batch/'profilemeta-provenance.json'
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
        print('已生成两项profiler元数据适配候选；需语义与性能验证', flush=True)


if __name__ == '__main__':
    main()
