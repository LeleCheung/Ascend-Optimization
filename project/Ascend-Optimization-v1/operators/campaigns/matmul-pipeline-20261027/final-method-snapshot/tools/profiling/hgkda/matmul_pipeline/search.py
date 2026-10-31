"""冻结候选空间，先小集合筛选，再对前两项扩展验证，不覆盖失败。"""
import argparse,hashlib,importlib.util,json,os,statistics,sys,time
from pathlib import Path
from templates import CONFIGS,generate,logical
HERE=Path(__file__).resolve().parent

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def save(path,d):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser()
    for name in ['parent','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--configs',nargs='+',required=True,choices=[c['id'] for c in CONFIGS]);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True);parent=a.parent.resolve()
    plan=dict(configs=[next(c for c in CONFIGS if c['id']==k) for k in a.configs],
        screening=[dict(size=1024,dtype='float16')],extension=[dict(size=s,dtype=d) for s in [1024,2048,4096] for d in ['float16','bfloat16']],
        repetitions=2,extend_top=2,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        selection='按同case实际父版／候选设备延迟，几何平均选；最终KGS前不得用完整合同结果继续调参',
        hypothesis='保留多个小累加块并调整二维复用、load-dot顺序与有限流水配置，避免已失败单大tile的UB与复用损失')
    save(out/'plan.json',plan);assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch,torch_npu
    sys.path.insert(0,str(HERE.parent/'mechanism'))
    helper=load(HERE.parent/'mechanism/probe.py','pipeline_capture')
    measure=load(HERE.parent/'device_mechanism/measure.py','pipeline_measure')
    compiler=load(HERE.parent/'device_mechanism/probe.py','pipeline_compiler')
    baseline=load(parent,'pipeline_baseline');state=dict(status='running',cases=[]);save(out/'state.json',state)
    def probe(row,candidate,spec):
        size,dt=spec['size'],spec['dtype'];directory=out/row['case']/(dt+'-'+str(size));directory.mkdir()
        gen=torch.Generator().manual_seed(20261027)
        def random(s):return torch.randn(s,generator=gen).to(dtype=getattr(torch,dt),device='npu')
        x,y,bias=random((size,size)),random((size,size)),random((size,))
        old,old_value=helper.capture_call(lambda _:baseline.run(x,y,bias),None)
        new,new_value=helper.capture_call(lambda _:candidate.run(x,y,bias),None)
        expected=torch.relu(x@y+bias);torch.npu.synchronize()
        torch.testing.assert_close(new_value,expected,atol=1e-4*size,rtol=1e-3 if dt=='float16' else .016)
        if len(old)!=1 or len(new)!=1:raise ValueError('不是单kernel路径')
        evidence=compiler.compiled_evidence(new,directory/'compiled');save(directory/'compiled.json',evidence)
        funcs=dict(parent=helper.direct_call(old),candidate=helper.direct_call(new));values={k:[] for k in funcs}
        for repeat in range(2):
            keys=list(funcs) if repeat==0 else list(reversed(funcs))
            times=measure.pair({k:funcs[k] for k in keys},dict(parent=old[0]['jit'].fn.__name__,candidate=new[0]['jit'].fn.__name__),directory/('timing-'+str(repeat)))
            for key in values:values[key].append(times[key])
        row['measurements'].append(dict(**spec,times=values,speedup=statistics.median(p/c for p,c in zip(values['parent'],values['candidate'])),
            logical=logical(row['config'],size,size,size)))
        save(out/'state.json',state)
    for cfg in plan['configs']:
        source=out/cfg['id']/'candidate.py';source.parent.mkdir();source.write_text(generate(parent.read_text(encoding='utf8'),cfg),encoding='utf8',newline='\n')
        row=dict(case=cfg['id'],config=cfg,status='running',measurements=[],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest());state['cases'].append(row);save(out/'state.json',state)
        try:
            candidate=load(source,'pipeline_'+cfg['id'].replace('-','_'));probe(row,candidate,plan['screening'][0]);row.update(status='screened',screen_speedup=row['measurements'][0]['speedup'])
        except Exception as error:row.update(status='failed',error=type(error).__name__+': '+str(error)[-5000:]);torch.npu.synchronize()
        save(out/'state.json',state);print(json.dumps({k:v for k,v in row.items() if k not in ['measurements','config']},ensure_ascii=False),flush=True)
    top=sorted([r for r in state['cases'] if r['status']=='screened'],key=lambda r:r['screen_speedup'],reverse=True)[:2]
    for row in top:
        try:
            source=out/row['case']/'candidate.py';candidate=load(source,'pipeline_extended_'+row['case'].replace('-','_'))
            for spec in plan['extension']:
                if spec==plan['screening'][0]:continue
                probe(row,candidate,spec)
            row.update(status='passed',speedup=statistics.geometric_mean(r['speedup'] for r in row['measurements']))
        except Exception as error:row.update(status='failed',error=type(error).__name__+': '+str(error)[-5000:]);torch.npu.synchronize()
        save(out/'state.json',state);print(json.dumps({k:v for k,v in row.items() if k not in ['measurements','config']},ensure_ascii=False),flush=True)
    valid=[r for r in top if r['status']=='passed'];best=max(valid,key=lambda r:r['speedup']) if valid else None
    state.update(status='completed',best=best['case'] if best else None,best_speedup=best['speedup'] if best else None);save(out/'state.json',state)

if __name__=='__main__':main()
