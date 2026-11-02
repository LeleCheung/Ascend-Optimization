"""独立严格Triton复核队列：锁内串行、固定合同、已提交请求不重发。"""
import argparse
import ast
import fcntl
import shlex
import time
from pathlib import Path
from common import BATCH,HERE,CANN,NEXT,PREVIOUS,ROOT,PY,read,save,sha,load
from gate import admit


class Runner:
    def __init__(self):
        self.module=load('strict_queue',HERE.parent/'run-nightly.py');self.q=self.module.Queue(BATCH.name,ROOT,'tle_yy')
        self.path=BATCH/'state.json';self.state=read(self.path) if self.path.exists() else dict(status='running',jobs={},operators={})
        self.state['status']='running';self.index=read(BATCH/'candidate-index.json')
        self.previous=read(PREVIOUS/'delivery.json');self.refs={r['operator']:r for r in self.previous['rows']}
    def persist(self):save(self.path,self.state)
    def wait(self):
        deadline=time.monotonic()+900
        while True:
            try:self.q.idle();return
            except self.module.Attention:
                if time.monotonic()>deadline:raise
                time.sleep(10)
    def job(self,key,fn):
        previous=self.state['jobs'].get(key)
        if previous and previous['status']=='completed':return previous.get('result')
        self.wait();self.state.update(current_job=key)
        self.state['jobs'][key]=dict(status='running',started_at=self.module.now());self.persist()
        print('START',key,flush=True)
        try:
            result=fn();self.state['jobs'][key].update(status='completed',result=result,finished_at=self.module.now())
            self.persist();print('DONE',key,flush=True);return result
        except Exception as e:
            self.state.update(status='requires_review',error=type(e).__name__+': '+str(e));self.persist();raise
    def cmd(self,key,args,budget=2100):
        return self.q.run(key,' '.join(shlex.quote(str(x)) for x in args),budget)
    def qualify(self,op,source,leaf):
        output=source.parent/'runtime-audit.json'
        if not output.exists():
            try:self.cmd('strict-runtime-'+op,[PY,HERE/'runtime_audit.py','--operator',op,'--source',source,'--leaf',leaf,'--output',output],1200)
            except RuntimeError:
                if not output.exists():raise
        value=read(output);assert value['source_sha256']==sha(source)
        return dict(status=value['status'],passed=value['passed'],total=value['total'],path=str(output))
    def formal(self,op,source,leaf,out,label):
        result=out/(label+'-1.result.json')
        self.cmd('formal-'+op+'-'+label,[PY,CANN/'evaluate.py','--operator',op,'--source',source,'--leaf',leaf,'--output',out,
            '--server',self.q.server,'--label',label,'--kind','strict_Triton_candidate'])
        v=read(result)
        return dict(status=v['status'],score=v.get('geo_mean'),passed=v.get('num_passed'),total=v.get('num_workloads'),
            is_hack=v.get('is_hack'),path=str(result),source_sha256=sha(source),measurement='new_once')
    def supplement(self,op,source,leaf,out,label):
        from validation_status import supplement_status
        if not out.exists():
            checker=HERE.parent/'reference_closure/validate_reference.py' if op=='_conj_physical' else CANN/'validate_cann.py'
            try:self.cmd(label+'-'+op,[PY,checker,'--operator',op,'--source',source,'--leaf',leaf,'--output',out],1200)
            except RuntimeError:
                if not out.exists():raise
        v=read(out)
        return dict(status=supplement_status(op,v,sha(source)),passed=v.get('passed'),total=v.get('total'),path=str(out),source_sha256=sha(source))
    def cann_leaf(self,op,row):
        old=BATCH/self.q.records[op]['catalog_leaf'];leaf=BATCH/'cann-catalog/ops'/op;leaf.mkdir(parents=True,exist_ok=True)
        for n in ['definition.json','correctness.jsonl','timing.jsonl']:
            p=leaf/n
            if p.exists():assert p.read_bytes()==(old/n).read_bytes()
            else:p.write_bytes((old/n).read_bytes())
        reference=Path(row['reference_source']);assert sha(reference)==row['reference_source_sha256']
        text=(old/'oracle.py').read_text(encoding='utf8')+'\n\n'+reference.read_text(encoding='utf8')
        p=leaf/'oracle.py'
        if p.exists():assert p.read_text(encoding='utf8')==text
        else:p.write_bytes(text.encode('utf8'))
        original=ast.parse((old/'oracle.py').read_text(encoding='utf8'));current=ast.parse(text)
        gen=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gen_inputs')
        assert ast.dump(gen(original))==ast.dump(gen(current))
        save(leaf.parents[1]/'manifest.json',dict(api_version='v6.2',name=BATCH.name+'-CANN',evaluator='native',layout='per-operator'))
        return leaf
    def trace(self,op,source,leaf):
        cases=[__import__('json').loads(x) for x in (leaf/'correctness.jsonl').read_text().splitlines() if x.strip()]
        case=cases[0];out=source.parent/'device-path';out.mkdir(parents=True,exist_ok=True)
        spec=out/'spec.json';payload=dict(operator=op,source=str(source),source_sha256=sha(source),leaf=str(leaf),case=case)
        if spec.exists():assert read(spec)==payload
        else:save(spec,payload)
        path=out/'trace.json'
        if not path.exists():self.cmd('strict-trace-'+op,[PY,NEXT/'driver.py','--spec',spec,'--side','candidate','--trace',path],600)
        parser=load('strict_trace_parser',HERE.parent/'threeway/trace_summary.py');summary=parser.summarize(path)
        save(out/'summary.json',summary)
        return dict(status='captured',trace=str(path),trace_sha256=sha(path),summary=str(out/'summary.json'),case=case['name'])
    def one(self,op):
        spec=self.index[op];source=Path(spec['source']);admit(source,source.parent/'review.json')
        leaf=BATCH/self.q.records[op]['catalog_leaf'];row=self.state['operators'].setdefault(op,{})
        row.update(source=str(source),source_sha256=sha(source))
        row['runtime']=self.job('runtime:'+op,lambda:self.qualify(op,source,leaf))
        if row['runtime']['status']!='passed':row['status']='运行时后端或执行失败，待修复';self.persist();return
        row['supplementary']=self.job('extra:'+op,lambda:self.supplement(op,source,leaf,source.parent/'extra-pytorch.json','extra-pytorch'))
        # 完整正确性及性能结果均保存；补测失败不能计作交付。
        row['pytorch']=self.job('pytorch:'+op,lambda:self.formal(op,source,leaf,source.parent/'pytorch','strict-PyTorch'))
        ref=self.refs[op]
        if ref.get('cann_reference_qualified'):
            alternate=self.cann_leaf(op,ref)
            previous=ref.get('starting_vs_cann') or {};old_gates=ref.get('starting_vs_cann_supplementary') or {}
            # 同字节候选、原输入/容差/参考与计时参数完全相同时复用原有效请求，避免重复挑分。
            old_source=PREVIOUS/'starting-points'/(op+'.py')
            if sha(source)==sha(old_source) and previous.get('passed') and old_gates.get('status')=='passed':
                row['cann']=dict(status='PASSED',score=previous['score'],passed=previous['complete'][0],total=previous['complete'][1],
                    path=previous['result'],source_sha256=sha(source),measurement='reused_exact_source_and_contract')
                row['cann_supplementary']=dict(old_gates,source_sha256=sha(source),measurement='reused_exact_source')
            else:
                row['cann_supplementary']=self.job('extra-cann:'+op,lambda:self.supplement(op,source,alternate,source.parent/'extra-cann.json','extra-CANN'))
                row['cann']=self.job('cann:'+op,lambda:self.formal(op,source,alternate,source.parent/'CANN','strict-CANN'))
            row['reference_kind']=ref['reference_kind'];row['reference_source']=ref['reference_source']
        else:row['reference_status']=ref['cann_status']
        if row['pytorch']['status']=='PASSED' and row['supplementary']['status']=='passed':
            row['device_path']=self.job('trace:'+op,lambda:self.trace(op,source,leaf))
        row['status']='本轮已复核';self.persist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--operator',action='append');a=p.parse_args()
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=Runner()
        for op in a.operator or list(r.index):
            r.one(op)
            load('strict_report',HERE/'report.py').main()
        r.wait();r.state.update(status='completed',current_job=None,finished_at=r.module.now());r.persist()
        load('strict_report',HERE/'report.py').main()

if __name__=='__main__':main()
