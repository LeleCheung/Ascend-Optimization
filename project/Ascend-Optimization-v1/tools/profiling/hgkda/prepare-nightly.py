#!/usr/bin/env python3
"""构建固定版本的15项夜间体检合同与原版候选；不修改上游快照。"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
REPO = PROJECT.parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


PARAMS = {
    'selu_': ['A'], 'rad2deg_': ['A'], 'log10_': ['A'],
    'value_selecting_reduction_backward': ['grad', 'dim', 'indices', 'sizes', 'keepdim'],
    'scatter_add': ['inp', 'dim', 'index', 'src'],
    'cumulative_trapezoid': ['y', 'dx', 'dim'],
    'uniform': ['self', 'from_', 'to'], 'conj_physical': ['input'],
    'upsample_bilinear2d': ['input', 'output_size', 'align_corners'],
    'fractional_max_pool3d': ['input', 'kernel_size', 'output_size', 'return_indices', '_random_samples'],
}


def cases(op):
    correctness, timing = [], []
    for dtype in ['float16', 'bfloat16', 'float32']:
        if op == 'value_selecting_reduction_backward':
            for sizes in [[3, 4], [4, 8], [2, 3, 5], [8, 16, 32]]:
                for dim in [0, 1, 2]:
                    for keep in [False, True]:
                        correctness.append(dict(dtype=dtype, sizes=sizes, dim=dim % len(sizes), keepdim=keep))
            correctness += [dict(dtype=dtype, sizes=[5, 7, 9], dim=-2, keepdim=keep, strided=True) for keep in [False, True]]
            timing += [dict(dtype=dtype, sizes=sizes, dim=len(sizes)-1, keepdim=False) for sizes in
                       [[1024, 1024], [4096, 4096], [8192, 8192], [32, 64, 128], [64, 128, 256], [128, 256, 512]]]
        elif op in {'selu_', 'rad2deg_', 'log10_'}:
            correctness += [dict(dtype=dtype, shape=shape, strided=strided) for shape in [[0], [1], [1025], [7, 9]] for strided in [False, True]]
            timing += [dict(dtype=dtype, shape=shape) for shape in [[1024], [65536], [1048576], [4096, 4096]]]
        elif op == 'scatter_add':
            correctness += [dict(dtype=dtype, shape=[5, 7, 9], dim=dim) for dim in [0, 1, 2]]
            timing += [dict(dtype=dtype, shape=shape, dim=-1) for shape in [[128, 128], [1024, 1024], [2048, 2048]]]
        elif op == 'cumulative_trapezoid':
            correctness += [dict(dtype=dtype, shape=shape, dim=dim, dx=dx, transpose=trans) for shape, dim, trans in
                            [([3, 17], -1, False), ([7, 9], 0, False), ([7, 9], -1, True), ([3, 1], -1, False)] for dx in [1, .25]]
            timing += [dict(dtype=dtype, shape=shape, dim=dim, dx=1) for shape, dim in [([128, 1024], -1), ([1024, 4096], -1), ([1024, 128], 0)]]
        elif op == 'uniform':
            correctness += [dict(dtype=dtype, shape=[65536], from_=low, to=high) for low, high in [(0., 1.), (-3., 2.), (.5, .5)]]
            timing += [dict(dtype=dtype, shape=[n], from_=0., to=1.) for n in [1024, 65536, 1048576]]
        elif op == 'upsample_bilinear2d':
            correctness += [dict(dtype=dtype, shape=[1, 3, 7, 9], output_size=out, align_corners=align) for out in [[13, 17], [1, 1], [3, 5]] for align in [True, False]]
            timing += [dict(dtype=dtype, shape=shape, output_size=out, align_corners=False) for shape, out in [([1, 16, 64, 64], [128, 128]), ([4, 32, 128, 128], [256, 256])]]
        elif op == 'fractional_max_pool3d':
            # 固定随机采样，同时核对value和indices。
            correctness += [dict(dtype=dtype, shape=[1, 2, 7, 9, 11], kernel_size=[2, 2, 2], output_size=[3, 4, 5])]
            timing += [dict(dtype=dtype, shape=shape, kernel_size=[2, 2, 2], output_size=out) for shape, out in [([1, 8, 16, 16, 16], [8, 8, 8]), ([2, 16, 32, 32, 32], [16, 16, 16])]]
    if op == 'conj_physical':
        for dtype in ['float32', 'complex64']:
            correctness += [dict(dtype=dtype, shape=[7, 9], transpose=trans, conj=conj) for trans in [True, False] for conj in [True, False]]
            timing += [dict(dtype=dtype, shape=[n]) for n in [1024, 65536, 1048576]]
    return correctness, timing


def native_signature(op, text):
    """保留上游默认值与keyword-only语义；仅SELU的可变参数需显式ABI入口。"""
    if op == 'selu_':
        return 'A', [{'name':'A','kind':'positional_or_keyword','required':True}]
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==op)
    args=copy.deepcopy(node.args)
    if args.vararg or args.kwarg:
        raise ValueError('需单独适配的可变参数: '+op)
    parameters=[]
    positional=args.posonlyargs+args.args
    defaults=[None]*(len(positional)-len(args.defaults))+list(args.defaults)
    for arg,default in list(zip(positional,defaults))+list(zip(args.kwonlyargs,args.kw_defaults)):
        arg.annotation=None
        kind='keyword_only' if arg in args.kwonlyargs else 'positional_only' if arg in args.posonlyargs else 'positional_or_keyword'
        row={'name':arg.arg,'kind':kind,'required':default is None}
        if default is not None:row['default']=ast.literal_eval(default)
        parameters.append(row)
    return ast.unparse(args),parameters


def build(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    catalog = output / 'catalog'
    records = []
    save(catalog / 'manifest.json', {'api_version': 'v6.2', 'name': 'hgkda-nightly-20261021', 'evaluator': 'native', 'layout': 'per-operator'})
    targets = json.loads((ROOT / 'targets.json').read_text(encoding='utf-8'))['targets']
    for target in targets:
        op = target['operator']
        leaf = catalog / 'ops' / op
        leaf.mkdir(parents=True)
        if target['source'] == 'flaggems_sglang':
            origin = REPO / target['native_contract_template']['contract_path']
            for name in ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']:
                shutil.copyfile(origin / name, leaf / name)
            definition = json.loads((leaf / 'definition.json').read_text(encoding='utf-8'))
            definition['name'] = op
            save(leaf / 'definition.json', definition)
            if op=='silu_and_mul_masked':
                oracle=(leaf/'oracle.py').read_text(encoding='utf-8')
                # KGS转换模板错误捕获了首个case的闭包；按上游_check_factory逐次绑定输入。
                prefix=oracle[:oracle.index('\nmasked_m = torch.tensor(')]
                helpers=oracle[oracle.index('\ndef assert_close('):oracle.index('\ndef valid(')]
                validator=(ROOT/'adapters/silu_masked_valid.py').read_text(encoding='utf-8')
                (leaf/'oracle.py').write_bytes((prefix+helpers+'\n'+validator).encode())
            source = ROOT / 'research/flaggems-sglang-9386120' / target['ascend_source']
            if not source.exists():
                raise FileNotFoundError('请先获取固定SGLang提交源码：' + str(source))
            text = source.read_text(encoding='utf-8') + '\nrun = ' + op + '\n'
            upstream = '938612058a981487e439da514df7d1371cf31e68'
            coverage = '固定KGS kernelcomp模板，尚需验证语义及覆盖；不是完整上游pytest测试集'
        else:
            params = PARAMS[op]
            name = 'selu_' if op == 'selu_' else 'rad2deg' if op == 'rad2deg_' else 'log10' if op == 'log10_' else op
            source = REPO / 'third_party/FlagGems/src/flag_gems/ops' / (name + '.py')
            upstream_text=source.read_text(encoding='utf-8')
            # 单文件候选没有父包上下文；仅修正导入地址，不改变原版计算。
            adapted_text=upstream_text.replace('from ..utils import dim_compress','from flag_gems.utils import dim_compress') if op=='scatter_add' else upstream_text
            signature,parameters=native_signature(op,upstream_text)
            effects = {'mutates': ['A'], 'returns_alias_of': {'out': 'A'}} if op in {'selu_', 'rad2deg_', 'log10_'} else {'mutates': [], 'returns_alias_of': {}}
            definition = {'api_version': 'v6.2', 'name': op, 'parameters': parameters,
                'outputs': ['out'], 'effects': effects}
            save(leaf / 'definition.json', definition)
            oracle = (ROOT / 'nightly_oracle.py').read_text(encoding='utf-8').replace('__OPERATOR__', op)
            if op != 'uniform':
                oracle = oracle[:oracle.index('\ndef valid(')]
            # Oracle保留明确ABI，避免**kwargs被原生合同加载器拒绝。
            oracle = oracle.replace('def run(**kwargs):', 'def run(' + signature + '):\n    kwargs = locals().copy()')
            (leaf / 'oracle.py').write_bytes(oracle.encode())
            correctness, timing = cases(op)
            for phase, specs in [('correctness', correctness), ('timing', timing)]:
                rows = [{'name': op + '-' + phase + '-' + str(i), 'inputs': {'spec': spec}, 'seed': 1729 + i,
                         'tolerance': {'atol': .015 if spec['dtype'] != 'float32' else .0001,
                                       'rtol': .015 if spec['dtype'] != 'float32' else .0001, 'required_matched_ratio': 1.0}} for i, spec in enumerate(specs)]
                (leaf / (phase + '.jsonl')).write_bytes(('\n'.join(json.dumps(x) for x in rows)+'\n').encode())
            if op == 'selu_':
                text=adapted_text+'\n_baseline_selu_ = selu_\ndef run(A):\n    return _baseline_selu_(A)\nselu_ = run\n'
            else:
                text=adapted_text+'\nrun = '+op+'\n'
            upstream = 'd6a8eec473517a3d68157b208eb9c057eb1d4c50'
            coverage = ('覆盖上游72项参数组合并加6项负维度/stride检查，性能使用原6形状×3dtype' if op == 'value_selecting_reduction_backward'
                        else '本机Native体检合同，覆盖代表性shape/dtype及特殊语义，不等同完整上游pytest')
        seed = output / 'seeds' / (op + '.py')
        seed.parent.mkdir(exist_ok=True)
        seed.write_bytes(text.encode())
        records.append({'operator': op, 'source': target['source'], 'upstream_commit': upstream,
                        'seed': seed.relative_to(output).as_posix(), 'catalog_leaf': leaf.relative_to(output).as_posix(),
                        'source_sha256': hashlib.sha256(seed.read_bytes()).hexdigest(),
                        'upstream_source': str(source.relative_to(REPO)).replace('.control-state/stage2-research/sglang/', ''),
                        'upstream_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'contract_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(leaf.iterdir()) if p.is_file()},
                        'correctness_cases': len((leaf/'correctness.jsonl').read_text().splitlines()),
                        'timing_cases': len((leaf/'timing.jsonl').read_text().splitlines()), 'coverage': coverage})
    save(output / 'plan.json', {'date': '2026-10-21', 'base_head': '72b199903c28dc7d61d9908367a5d04feea09fdf',
        'timing_scope': 'walltime', 'server': 'http://127.0.0.1:19659', 'physical_card': 7,
        'arms': ['FlagGems', 'kg-no-profile', 'kg-profile', 'HG-KDA'], 'records': records})
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps({'operators': len(build(a.output))}))
