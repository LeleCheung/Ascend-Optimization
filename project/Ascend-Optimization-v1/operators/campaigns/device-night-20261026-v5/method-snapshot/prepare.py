"""冻结设备机制起点、开发/独立合同及干预空间；不读取独立集结果。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from variants import probes

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    if out.exists():raise ValueError('请用新批次')
    records=[]
    for op in ['matmul_bias_activation','amin']:
        original=PROJECT/'operators'/op/'reports/profiling-agent-20261014-four-refined-method/final.py'
        seed=out/'seeds'/(op+'.py');seed.parent.mkdir(parents=True,exist_ok=True)
        source=original.read_text(encoding='utf8')
        compatibility=[]
        if op=='matmul_bias_activation':
            old='multibuffer=not tiny, unit_flag=not tiny, sync_solver=False,'
            assert source.count(old)==1
            source=source.replace(old,'multibuffer=not tiny and k >= 2*bk, unit_flag=not tiny and k >= 2*bk, sync_solver=False,',1)
            compatibility=['pipeline路径单K迭代关闭multibuffer/unit_flag；两组共享，原父版aicore超时证据保留']
        seed.write_bytes(source.encode())
        leaves={}
        for split in ['development','heldout']:
            leaf=out/'catalogs'/split/'ops'/op;leaf.mkdir(parents=True)
            params=['input','weight','bias'] if op=='matmul_bias_activation' else ['inp','dim','keepdim']
            definition=dict(api_version='v6.2',name=op,parameters=[dict(name=n,kind='positional_or_keyword',required=True) for n in params],outputs=['out'])
            save(leaf/'definition.json',definition)
            oracle=(HERE/'oracle.py').read_text(encoding='utf8').replace('__OPERATOR__',op).replace('def run(**inputs):','def _reference(**inputs):')
            oracle+= ('\ndef run(input,weight,bias):\n    return _reference(input=input,weight=weight,bias=bias)\n' if op=='matmul_bias_activation' else
                      '\ndef run(inp,dim,keepdim):\n    return _reference(inp=inp,dim=dim,keepdim=keepdim)\n')
            (leaf/'oracle.py').write_bytes(oracle.encode())
            for phase in ['correctness','timing']:
                rows=[]
                if op=='matmul_bias_activation':
                    shapes=([[256,256,256],[1024,1024,1024]] if split=='development' else [[768,768,768],[1536,1536,1536]]) if phase=='timing' else ([[31,47,65],[128,256,128],[256,256,256]] if split=='development' else [[33,63,127],[192,256,128],[384,512,256]])
                else:
                    shapes=([[4096,4096],[64,512,512]] if split=='development' else [[3072,3072],[48,384,768]]) if phase=='timing' else ([[17,33],[4,65,33],[128,256]] if split=='development' else [[19,47],[3,67,39],[192,384]])
                for dtype in ['float16','bfloat16','float32']:
                    for shape in shapes:
                        for layout in ([False,True] if phase=='correctness' else [False]):
                            spec=dict(shape=shape,dtype=dtype,strided=layout)
                            if op=='amin':spec.update(dim=1,keepdim=layout)
                            else:spec.update(bias_row=layout)
                            rows.append(dict(name=op+'-'+split+'-'+phase+'-'+str(len(rows)),inputs={'spec':spec},seed=71329+len(rows),
                                tolerance=dict(atol=1e-4*shape[-1] if op=='matmul_bias_activation' else 0,
                                    rtol={'float16':1e-3,'bfloat16':.016,'float32':1.3e-6}[dtype] if op=='matmul_bias_activation' else 0,required_matched_ratio=1.0)))
                (leaf/(phase+'.jsonl')).write_bytes(('\n'.join(json.dumps(r) for r in rows)+'\n').encode())
            leaves[split]=dict(path=leaf.relative_to(out).as_posix(),sha256={f.name:sha(f) for f in leaf.iterdir()})
        configs=[]
        for name,cfg in probes(op):
            cfgfile=out/'probe-configs'/op/(name+'.json');save(cfgfile,cfg)
            configs.append(dict(name=name,config=cfg,path=cfgfile.relative_to(out).as_posix()))
        records.append(dict(operator=op,seed=seed.relative_to(out).as_posix(),source_sha256=sha(seed),
            catalog_leaf=leaves['development']['path'],contract_sha256=leaves['development']['sha256'],
            splits=leaves,probes=configs,origin=str(original.relative_to(PROJECT)),
            source_role='历史KG/HG-KDA/CC与Codex参与优化版＋共享兼容修复；作为本轮父版，不冒充原版FlagGems',
            compatibility=compatibility,original_source_sha256=sha(original)))
    snapshot=out/'method-snapshot';snapshot.mkdir()
    files=[p for p in HERE.glob('*.py')]+[HERE/'README.md']
    for path in files:
        if path.exists():shutil.copyfile(path,snapshot/path.name)
    shutil.copyfile(HERE.parent/'mechanism/probe.py',snapshot/'launch-probe.py')
    shutil.copyfile(HERE.parent/'mechanism/core.py',snapshot/'core.py')
    save(out/'plan.json',dict(batch=out.name,server='http://127.0.0.1:19659',physical_card=7,
        model='deepseek-flash',arms=['raw_evidence','mechanism'],candidates_per_arm=2,format_repairs=1,
        formal_repetitions=1,records=records,method_sha256={p.name:sha(p) for p in snapshot.iterdir()},
        policy='两组共用相同干预配置空间和观测；方法组增加机制诊断；独立合同预登记，冻结后才测量。'))
    print(json.dumps(dict(batch=str(out),operators=len(records)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
