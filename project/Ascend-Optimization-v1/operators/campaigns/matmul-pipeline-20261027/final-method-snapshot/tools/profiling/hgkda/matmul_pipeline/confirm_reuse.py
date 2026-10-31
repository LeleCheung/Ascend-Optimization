"""冻结后的同模板复用方向确认；只解释机制，不用于重新选版。"""
import argparse,hashlib,importlib.util,json,os,statistics,sys
from pathlib import Path
from templates import CONFIGS,generate
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve();out=b/'matched-confirmation';out.mkdir()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    parent=PROJECT/'operators/campaigns/roofline-20261027/validation/delivered.py'
    c={r['id']:r for r in CONFIGS};configs=[c['dual-n-constexpr'],c['dual-m-reuse-b']]
    save(out/'plan.json',dict(configs=configs,sizes=[1024,4096],dtypes=['float16','bfloat16'],repetitions=2,
        policy='正式选版已冻结；对比相同模板下复用方向，避免将形状constexpr等联合变化全归为B复用收益'))
    import torch,torch_npu
    sys.path.insert(0,str(HERE.parent/'mechanism'))
    helper=load(HERE.parent/'mechanism/probe.py','confirm_capture');measure=load(HERE.parent/'device_mechanism/measure.py','confirm_measure')
    compiler=load(HERE.parent/'device_mechanism/probe.py','confirm_compiler')
    modules={};sources={}
    for config in configs:
        path=out/(config['id']+'.py');path.write_text(generate(parent.read_text(encoding='utf8'),config),encoding='utf8',newline='\n')
        sources[config['id']]=hashlib.sha256(path.read_bytes()).hexdigest();modules[config['id']]=load(path,'confirm_'+config['id'].replace('-','_'))
    state=dict(status='running',source_sha256=sources,cases=[]);save(out/'state.json',state)
    for size in [1024,4096]:
        for dt in ['float16','bfloat16']:
            gen=torch.Generator().manual_seed(27171+size)
            def random(s):return torch.randn(s,generator=gen).to(dtype=getattr(torch,dt),device='npu')
            x,w,bias=random((size,size)),random((size,size)),random((size,));ref=torch.relu(x@w+bias)
            root=out/(dt+'-'+str(size));root.mkdir();captured={};functions={};symbols={}
            for name,m in modules.items():
                calls,result=helper.capture_call(lambda _:m.run(x,w,bias),None)
                torch.npu.synchronize();torch.testing.assert_close(result,ref,atol=1e-4*size,rtol=1e-3 if dt=='float16' else .016)
                if len(calls)!=1:raise ValueError('确认不是单kernel')
                captured[name]=compiler.compiled_evidence(calls,root/name/'compiled');functions[name]=helper.direct_call(calls);symbols[name]=calls[0]['jit'].fn.__name__
            values={key:[] for key in functions}
            for repeat in [0,1]:
                keys=list(functions) if repeat==0 else list(reversed(functions))
                result=measure.pair({key:functions[key] for key in keys},symbols,root/('timing-'+str(repeat)))
                for key in values:values[key].append(result[key])
            row=dict(size=size,dtype=dt,times=values,m_reuse_over_n_reuse=statistics.median(p/c for p,c in zip(values['dual-n-constexpr'],values['dual-m-reuse-b'])),compiled=captured)
            state['cases'].append(row);save(out/'state.json',state)
    state.update(status='completed',geomean=statistics.geometric_mean(r['m_reuse_over_n_reuse'] for r in state['cases']))
    save(out/'state.json',state);print(json.dumps(dict(status='completed',geomean=state['geomean'])))

if __name__=='__main__':main()
