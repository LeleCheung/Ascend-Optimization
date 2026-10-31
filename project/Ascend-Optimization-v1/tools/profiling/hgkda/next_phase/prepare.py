"""冻结导师新25项目标、源码、Native合同与夜间队列；不修改上游快照。"""
import argparse
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
REPO = PROJECT.parents[1]
TARGETS = 'negative_ selu_ xlogy_scalar_tensor nan_to_num_ cross_out expm1_ hardshrink_backward subtract_ native_group_norm log2_ rad2deg_ transform_bias_rescale_qkv fix log10_ round_out cumulative_trapezoid hardswish_out fake_quantize_per_tensor_affine_cachemask_out special_round_out special_xlogy_ hardswish fmod_tensor_ uniform linalg_inv_ex batch_norm_with_update_functional'.split()
ENTRIES={'transform_bias_rescale_qkv':'_transform_bias_rescale_qkv','batch_norm_with_update_functional':'_batch_norm_with_update_functional'}
HISTORICAL = {
    'selu_': ('operators/campaigns/buffer-pointwise-20261022', None),
    'rad2deg_': ('operators/campaigns/night-20261021-v3', None),
    'log10_': ('operators/campaigns/buffer-pointwise-20261022', None),
    'uniform': ('operators/campaigns/night-20261021-v3', None),
    'cumulative_trapezoid': ('operators/campaigns/night-20261021-v3', None),
}
STARTING_POINTS={
    'selu_':'operators/campaigns/endurance-20261022/deliverables/selu_/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0/main.py',
    'rad2deg_':'operators/campaigns/endurance-20261022/deliverables/rad2deg_/3b2c0328b1857ef802e575805c4e25d002ac0166c22fa5cc6804cb1c8d0eeecc/main.py',
    'uniform':'operators/campaigns/endurance-20261022/deliverables/uniform/b4c0ba819fe77ead84221a61bc8219d1655c870d3a40e408a3a4fcaeee87f7ea/main.py',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def signature(node):
    args = copy.deepcopy(node.args)
    if args.vararg or args.kwarg:
        if node.name != 'selu_':
            raise ValueError('需要单独适配可变参数 '+node.name)
        args = ast.parse('def run(A): pass').body[0].args
    parameters = []
    pos = args.posonlyargs + args.args
    defaults = [None]*(len(pos)-len(args.defaults)) + list(args.defaults)
    for arg, default in list(zip(pos, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults)):
        arg.annotation = None
        row = dict(name=arg.arg, kind='keyword_only' if arg in args.kwonlyargs else 'positional_only' if arg in args.posonlyargs else 'positional_or_keyword', required=default is None)
        if default is not None:
            row['default'] = ast.literal_eval(default)
        parameters.append(row)
    return ast.unparse(args), parameters


def cases(op):
    correct, timing = [], []
    dtypes = ['float16', 'bfloat16', 'float32'] if op != 'linalg_inv_ex' else ['float32']
    for dtype in dtypes:
        if op == 'transform_bias_rescale_qkv':
            correct += [dict(dtype=dtype,shape=s,heads=h,strided=st) for s,h in [([1,8,192],4),([2,17,384],8)] for st in [False,True]]
            timing += [dict(dtype=dtype,shape=s,heads=h) for s,h in [([2,128,768],8),([4,1024,1536],8),([8,512,3072],16)]]
        elif op == 'batch_norm_with_update_functional':
            correct += [dict(dtype=dtype,shape=s,affine=af) for s in [[4,3],[2,8,7,9],[2,16,3,5,7]] for af in [True,False]]
            timing += [dict(dtype=dtype,shape=s,affine=True) for s in [[4,32,32,32],[8,64,64,64],[16,128,32,32]]]
        elif op == 'native_group_norm':
            shapes = [[1, 8, 3, 5], [2, 16, 7, 9], [2, 32, 1, 1]]
            for shape in shapes:
                for affine in [True, False]:
                    correct.append(dict(dtype=dtype, shape=shape, groups=4, affine=affine))
            timing += [dict(dtype=dtype, shape=s, groups=8, affine=True) for s in [[4, 32, 32, 32], [8, 64, 64, 64], [16, 128, 32, 32]]]
        elif op == 'linalg_inv_ex':
            correct += [dict(dtype=dtype, shape=s, singular=bad) for s in [[3, 3], [2, 7, 7]] for bad in [False, True]]
            timing += [dict(dtype=dtype, shape=s) for s in [[16, 16], [8, 32, 32], [16, 64, 64]]]
        elif op == 'cross_out':
            correct += [dict(dtype=dtype, shape=s, dim=dim, strided=st) for s, dim in [([3, 7], 0), ([2, 3, 9], 1)] for st in [False, True]]
            timing += [dict(dtype=dtype, shape=s, dim=-1) for s in [[1024, 3], [65536, 3], [1048576, 3]]]
        else:
            correct += [dict(dtype=dtype, shape=s, strided=st, special=True) for s in [[1], [1025], [7, 9]] for st in [False, True]]
            timing += [dict(dtype=dtype, shape=s) for s in [[1024], [65536], [1048576], [4096, 4096]]]
    return correct, timing


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    functions = {}
    for path in sorted((REPO/'third_party/FlagGems/src/flag_gems/ops').glob('*.py')):
        text = path.read_text(encoding='utf8')
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in TARGETS:
                functions[node.name] = (path, node, text)
            elif isinstance(node,ast.FunctionDef) and node.name in ENTRIES.values():
                functions[next(k for k,v in ENTRIES.items() if v==node.name)]=(path,node,text)
    records, targets = [], []
    save(output/'catalog/manifest.json', dict(api_version='v6.2', name=output.name, evaluator='native', layout='per-operator'))
    for order, op in enumerate(TARGETS, 1):
        target = dict(order=order, operator=op, repository='FlagGems', exact_adapter_definition=(REPO/'sources/kernelgen_server/data/flaggems-adapter-definitions/definitions'/(op+'.json')).is_file())
        targets.append(target)
        if op not in functions:
            target.update(status='needs_interface_mapping', reason='固定FlagGems中无同名入口，须核对带下划线内部接口及PyTorch schema；不静默改名')
            continue
        path, node, text = functions[op]
        if op in ENTRIES:
            target['interface_mapping']=dict(requested_name=op,actual_flaggems_entry=ENTRIES[op],actual_pytorch_entry='aten.'+ENTRIES[op],
                evidence=['third_party/FlagGems/tests/test_'+op+'.py',path.relative_to(REPO).as_posix()],status='源码与上游测试调用对应，等待本机真实合同验证')
        sig, parameters = signature(node)
        leaf = output/'catalog/ops'/op
        leaf.mkdir(parents=True)
        seed = output/'seeds'/(op+'.py')
        seed.parent.mkdir(exist_ok=True)
        # 保持计算不变，仅将单文件内相对导入改为固定包的绝对导入。
        text = text.replace('from ..utils import', 'from flag_gems.utils import')
        text = text.replace('from .', 'from flag_gems.ops.')
        if op == 'selu_':
            text += '\n_original_selu = selu_\ndef run(A):\n    return _original_selu(A)\n'
        else:
            text += '\nrun = '+ENTRIES.get(op,op)+'\n'
        seed.write_text(text, encoding='utf8')
        if op in HISTORICAL:
            old_dir = PROJECT/HISTORICAL[op][0]
            old_plan = old_dir/'plan.json'
            if old_plan.exists():
                old = next((r for r in json.loads(old_plan.read_text(encoding='utf8'))['records'] if r['operator']==op), None)
                if old:
                    for name in ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']:
                        shutil.copyfile(old_dir/old['catalog_leaf']/name, leaf/name)
                    target['contract_origin'] = old_dir.relative_to(PROJECT).as_posix()+'/'+old['catalog_leaf']
        if not (leaf/'definition.json').exists():
            outputs = ['q','k','v'] if op=='transform_bias_rescale_qkv' else ['out','save_mean','save_invstd','reserve','running_mean_out','running_var_out'] if op=='batch_norm_with_update_functional' else ['out', 'mean', 'rstd'] if op=='native_group_norm' else ['inverse', 'info'] if op=='linalg_inv_ex' else ['out0', 'out1'] if op=='fake_quantize_per_tensor_affine_cachemask_out' else ['out']
            aliases = {}
            if op.endswith('_'):
                aliases['out'] = parameters[0]['name']
            if op.endswith('_out'):
                aliases = {'out0':'out0','out1':'out1'} if op.startswith('fake_quantize') else {'out':'out'}
            save(leaf/'definition.json', dict(api_version='v6.2', name=op, parameters=parameters, outputs=outputs,
                 effects=dict(mutates=list(aliases.values()), returns_alias_of=aliases)))
            oracle = (HERE/'oracle.py').read_text(encoding='utf8').replace('__OPERATOR__', op)
            oracle = oracle.replace('def run(**kwargs):', 'def run('+sig+'):\n    kwargs = locals().copy()')
            (leaf/'oracle.py').write_text(oracle, encoding='utf8')
            for phase, specs in zip(['correctness', 'timing'], cases(op)):
                rows = [dict(name=op+'-'+phase+'-'+str(i), inputs=dict(spec=spec), seed=1729+i,
                             tolerance=dict(atol=0.015 if spec['dtype']!='float32' else 0.0001, rtol=0.015 if spec['dtype']!='float32' else 0.0001, required_matched_ratio=1.0, equal_nan=True)) for i,spec in enumerate(specs)]
                (leaf/(phase+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows)+'\n', encoding='utf8')
        record = dict(operator=op, seed=seed.relative_to(output).as_posix(), catalog_leaf=leaf.relative_to(output).as_posix(), source_sha256=sha(seed),
                      upstream_source=path.relative_to(REPO).as_posix(), upstream_source_sha256=sha(path),
                      contract_sha256={p.name:sha(p) for p in leaf.iterdir()},
                      coverage='独立Native代表合同；特殊值/stride/out需补充检查，不冒充上游完整pytest')
        if op in ENTRIES:record['actual_entry']=ENTRIES[op]
        if op in STARTING_POINTS and (PROJECT/STARTING_POINTS[op]).is_file():
            original=PROJECT/STARTING_POINTS[op];start=output/'starting-points'/(op+'.py');start.parent.mkdir(exist_ok=True);shutil.copyfile(original,start)
            record.update(starting_source=start.relative_to(output).as_posix(),starting_source_sha256=sha(start),starting_source_origin=STARTING_POINTS[op])
        records.append(record)
        target.update(status='contract_prepared_pending_validation', **record)
    save(output/'targets.json', dict(schema='hgkda-next25-v1', targets=targets))
    plan = dict(schema_version='hgkda-next-phase-v1', server='http://127.0.0.1:19659', physical_card=7,
                timing_scope='walltime', final_repeats=1, duration_hours=14, records=records,
                priority=['expm1_', 'native_group_norm', 'negative_', 'nan_to_num_', 'log2_', 'hardshrink_backward'],
                rounds=3, passes=[dict(rounds=3,budget_seconds=2400)], profile_budget_seconds=180,
                policy='既有流程先运行，新方法在代表项同步接入；一次完整复验；失败保留并转下一项；准确记录Codex贡献')
    save(output/'plan.json', plan)
    save(output/'software-lock.json', json.loads((REPO/'manifests/hgkda-stage2.lock.json').read_text(encoding='utf8')))
    return dict(targets=len(targets), prepared=len(records), needs_mapping=[t['operator'] for t in targets if t['status']=='needs_interface_mapping'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().output), ensure_ascii=False))
