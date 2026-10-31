"""创建独立算子合同与固定上游源码，方法冻结先于独立检验。"""
from pathlib import Path
import argparse
import json
import hashlib
import shutil

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
REPO=PROJECT.parents[1]

ORACLE='''"""独立PyTorch参考及CPU随机输入，不被候选导入。"""
import torch
REFERENCE_DEVICE='target'
OPERATOR=__OPERATOR__
def gen_inputs(ctx, device):
    s=ctx['inputs']['spec']
    g=torch.Generator().manual_seed(ctx.get('seed',1729))
    x=torch.randn(s['shape'],generator=g).to(dtype=getattr(torch,s['dtype']),device=device)
    if s.get('strided'):
        base=torch.empty((*x.shape,2),dtype=x.dtype,device=device)
        base[...,0]=x
        x=base[...,0]
    return dict(A=x)
def run(A):
    return getattr(torch,OPERATOR)(A)
'''


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(output):
    if output.exists():raise ValueError('批次存在，不覆盖')
    output.mkdir(parents=True)
    catalog=output/'catalog'
    catalog.mkdir()
    (catalog/'manifest.json').write_text(json.dumps(dict(api_version='v6.2',name=output.name,evaluator='native',layout='per-operator')),encoding='utf8')
    records=[]
    for op,role in [('selu_','development'),('sigmoid_','heldout_operator')]:
        leaf=catalog/'ops'/op;leaf.mkdir(parents=True)
        definition=dict(api_version='v6.2',name=op,parameters=[dict(name='A',kind='positional_or_keyword',required=True)],
                        outputs=['out'],effects=dict(mutates=['A'],returns_alias_of={'out':'A'}))
        (leaf/'definition.json').write_text(json.dumps(definition,indent=2),encoding='utf8')
        (leaf/'oracle.py').write_text(ORACLE.replace('__OPERATOR__',repr(op)),encoding='utf8')
        # 两算子使用同分布、但不同形状；独立检验shape不参与开发调参。
        shapes=[[],[0],[1],[17],[1025],[7,9],[2,3,5]]
        timing=[[1024],[65536],[1048576]] if role=='development' else [[1537],[98304],[1572864]]
        for phase in ['correctness','timing']:
            rows=[]
            for dtype in ['float16','bfloat16','float32']:
                for shape in shapes if phase=='correctness' else timing:
                    for strided in [False,True] if phase=='correctness' else [False]:
                        rows.append(dict(name=op+'-'+phase+'-'+str(len(rows)),inputs={'spec':dict(shape=shape,dtype=dtype,strided=strided)},
                            seed=1729+len(rows),tolerance=dict(atol=1e-4 if dtype=='float32' else .015,
                            rtol=1e-4 if dtype=='float32' else .015,required_matched_ratio=1.0)))
            (leaf/(phase+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf8')
        upstream=REPO/'third_party/FlagGems/src/flag_gems/ops'/('selu_.py' if op=='selu_' else 'sigmoid.py')
        seed=output/'seeds'/(op+'.py');seed.parent.mkdir(exist_ok=True)
        suffix='\n_baseline=selu_\ndef run(A):\n    return _baseline(A)\nselu_=run\n' if op=='selu_' else '\nrun=sigmoid_\n'
        seed.write_bytes((upstream.read_text(encoding='utf8')+suffix).encode())
        records.append(dict(operator=op,role=role,seed=seed.relative_to(output).as_posix(),source_sha256=digest(seed),
            upstream_source=upstream.relative_to(REPO).as_posix(),upstream_sha256=digest(upstream),
            catalog_leaf=leaf.relative_to(output).as_posix(),contract_sha256={p.name:digest(p) for p in leaf.iterdir()}))
    tools={p.name:digest(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md']}
    plan=dict(batch=output.name,server='http://127.0.0.1:19659',physical_card=7,
              method='launch-mechanism-v1',scope='原型验证，不是完整指令依赖图或通用反事实引擎',
              source_commit='d6a8eec473517a3d68157b208eb9c057eb1d4c50',
              records=records,tools_sha256=tools,model='deepseek-flash',
              arms=['raw_evidence','mechanism'],candidate_calls_per_arm=1,format_repairs=1,
              predeclared_dispatch_ratio=1.15,formal_repetitions=1,
              split='selu_开发；冻结方法后sigmoid_独立算子与不同形状检验；不将历史最佳代码提供给模型')
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    freeze=output/'method-snapshot';freeze.mkdir()
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ['.py','.md']:shutil.copyfile(p,freeze/p.name)
    print(json.dumps(dict(output=str(output),records=len(records)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    prepare(p.parse_args().output)
