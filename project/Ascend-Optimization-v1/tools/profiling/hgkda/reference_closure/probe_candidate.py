"""冻结合同全输入诊断：独立进程报告inverse/info/残差，不替代正式KGS。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser()
    for k in ['source','leaf','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch,torch_npu
    oracle=load('inverse_probe_oracle',a.leaf/'oracle.py');candidate=load('inverse_probe_candidate',a.source);rows=[]
    for phase in ['correctness','timing']:
        for case in [json.loads(t) for t in (a.leaf/(phase+'.jsonl')).read_text().splitlines() if t.strip()]:
            row=dict(case=case['name'],phase=phase,passed=False)
            try:
                inputs=oracle.gen_inputs(case,'npu:0');refargs=oracle.gen_inputs(case,'npu:0');matrix=inputs['A'].cpu()
                ref,info=oracle.run(**refargs);value,got_info=candidate.run(**inputs);torch.npu.synchronize()
                ref,info,value,got_info=[x.cpu() for x in [ref,info,value,got_info]]
                row.update(finite=bool(torch.isfinite(value).all()),reference_info=info.tolist(),candidate_info=got_info.tolist(),
                    info_matched=bool(torch.equal(info,got_info)),inverse_matched=bool(torch.allclose(value,ref,atol=case['tolerance']['atol'],rtol=case['tolerance']['rtol'],equal_nan=True)))
                residual=matrix.double()@value.double()-torch.eye(matrix.shape[-1],dtype=torch.float64)
                row['residual_max']=float(residual.abs().max()) if torch.isfinite(residual).all() else None
                diff=(value-ref).abs();row['max_absolute_difference']=float(diff.max()) if torch.isfinite(diff).all() else None
                row['passed']=row['inverse_matched'] and row['info_matched']
                if matrix.shape[-1]<=7:row.update(input=matrix.tolist(),candidate=value.tolist(),reference=ref.tolist())
            except Exception as e:row['error']=type(e).__name__+': '+str(e)[:2000]
            rows.append(row)
    def clean(v):
        if isinstance(v,float) and not __import__('math').isfinite(v):return str(v)
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        return v
    result=clean(dict(status='diagnostic_passed' if all(r['passed'] for r in rows) else 'diagnostic_failed',source=str(a.source),source_sha256=sha(a.source),probe_sha256=sha(__file__),
        passed=sum(r['passed'] for r in rows),total=len(rows),cases=rows,contract_sha256={n:sha(a.leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']},scope='一次全输入诊断；无性能成绩，正式评测另行执行'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
if __name__=='__main__':main()
