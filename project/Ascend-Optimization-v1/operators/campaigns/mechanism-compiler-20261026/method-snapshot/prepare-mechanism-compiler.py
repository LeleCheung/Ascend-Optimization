"""冻结机制到代码变换的修订实验；独立算子使用tanh_，布局修复两组共享。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2];REPO=PROJECT.parents[1]
spec=importlib.util.spec_from_file_location('mechanism_prepare',HERE/'mechanism/prepare.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(out):
    if out.exists():raise ValueError('保留旧批次，请用新目录')
    out.mkdir(parents=True);records=[]
    catalog=out/'catalog';catalog.mkdir()
    (catalog/'manifest.json').write_text(json.dumps(dict(api_version='v6.2',name=out.name,evaluator='native',layout='per-operator')))
    for op,role in [('selu_','development'),('tanh_','heldout_operator')]:
        leaf=catalog/'ops'/op;leaf.mkdir(parents=True)
        definition=dict(api_version='v6.2',name=op,parameters=[dict(name='A',kind='positional_or_keyword',required=True)],
                        outputs=['out'],effects=dict(mutates=['A'],returns_alias_of={'out':'A'}))
        (leaf/'definition.json').write_text(json.dumps(definition,indent=2))
        (leaf/'oracle.py').write_bytes(base.ORACLE.replace('__OPERATOR__',repr(op)).encode())
        shapes=[[],[0],[1],[17],[1025],[7,9],[2,3,5]]
        timing=[[1024],[65536],[1048576]] if role=='development' else [[2305],[81920],[1310720]]
        for phase in ['correctness','timing']:
            rows=[]
            for dtype in ['float16','bfloat16','float32']:
                for shape in shapes if phase=='correctness' else timing:
                    for strided in [False,True] if phase=='correctness' else [False]:
                        rows.append(dict(name=op+'-'+phase+'-'+str(len(rows)),inputs={'spec':dict(shape=shape,dtype=dtype,strided=strided)},
                            seed=61729+len(rows),tolerance=dict(atol=1e-4 if dtype=='float32' else .015,
                            rtol=1e-4 if dtype=='float32' else .015,required_matched_ratio=1.0)))
            (leaf/(phase+'.jsonl')).write_bytes(('\n'.join(json.dumps(r) for r in rows)+'\n').encode())
        original=REPO/'third_party/FlagGems/src/flag_gems/ops'/('selu_.py' if op=='selu_' else 'tanh.py')
        text=original.read_text(encoding='utf8')
        # 两组相同的显式布局适配：不改变数学公式，也不将修复收益混入优化。
        adapter='\n_hg_layout_parent = '+op+'''
def run(A):
    if not A.is_contiguous():
        temporary = A.contiguous()
        _hg_layout_parent(temporary)
        A.copy_(temporary)
        return A
    return _hg_layout_parent(A)
'''+op+' = run\n'
        seed=out/'seeds'/(op+'.py');seed.parent.mkdir(exist_ok=True);seed.write_bytes((text+adapter).encode())
        upstream=out/'upstream'/(op+'.py');upstream.parent.mkdir(exist_ok=True);upstream.write_bytes(original.read_bytes())
        records.append(dict(operator=op,role=role,seed=seed.relative_to(out).as_posix(),source_sha256=sha(seed),
            upstream_sha256=sha(original),baseline='固定FlagGems数学实现＋显式共享布局适配',
            catalog_leaf=leaf.relative_to(out).as_posix(),contract_sha256={p.name:sha(p) for p in leaf.iterdir()}))
    freeze=out/'method-snapshot';freeze.mkdir()
    files=list((HERE/'mechanism').glob('*.py'))+list((HERE/'mechanism').glob('*.md'))
    files += [HERE/'rewrite-dispatch.py',HERE/'run-mechanism-compiler.py',Path(__file__)]
    hashes={}
    for p in files:shutil.copyfile(p,freeze/p.name);hashes[p.name]=sha(p)
    plan=dict(batch=out.name,server='http://127.0.0.1:19659',physical_card=7,
        records=records,model='deepseek-flash',arms=['raw_evidence','mechanism_rewrite'],
        calls_per_arm=1,format_repairs=1,formal_repetitions=1,method_sha256=hashes,
        change_reason='v2诊断正确但自由生成代码重复执行；新增有前提的通用变换，不手修模型候选。sigmoid原版48/51失败保留；独立改用新tanh_及新形状，并显式共享布局适配。',
        split='冻结前仅SELU开发；tanh_不参与规则与变换开发，独立算子使用不同shape',
        comparison='原始证据自由源码生成 vs 机制诊断＋受约束变换；组合系统对照，不能单独归因执行图')
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(batch=str(out),records=len(records)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);prepare(p.parse_args().output)
