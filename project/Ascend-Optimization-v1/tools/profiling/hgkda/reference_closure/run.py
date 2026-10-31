"""tmux串行夜间闭环：资格、原合同、命名NPU参考、路径及KG补充优化。"""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
NEXT=HERE.parent/'next_phase';THREE=HERE.parent/'threeway';sys.path.insert(0,str(NEXT))
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('reference_closure_next',NEXT/'run.py')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quote(parts):return ' '.join(shlex.quote(str(x)) for x in parts)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
class Runner(base.Runner):
    def save(self):
        base.save_json(self.path,self.state)
        lines=['# 夜间六项与CANN参考进展','','常规稳定达标起点24/30；新参考资格与候选成果分别列示。','','|任务|状态|结果|','|---|---|---|']
        for key,v in self.state.get('jobs',{}).items():
            result=v.get('result');detail=''
            if isinstance(result,dict):detail=str({k:result[k] for k in ['status','geo_mean','passed','total','eligible_full_contract','error'] if k in result})
            lines.append('|'+key+'|'+v['status']+'|'+detail.replace('|','/')[:500]+'|')
        (self.batch/'进展.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    def _cmd(self,key,args,budget=2100):self.q.run(key,quote(args),budget)
    def supplemental_at(self,op,source,gates,label,leaf):
        from validation_status import supplement_status
        if not gates.exists():
            try:
                self._cmd(label,[base.PY,NEXT/'validate.py','--operator',op,'--source',source,
                    '--leaf',leaf,'--output',gates],1200)
            except RuntimeError:
                if not gates.exists():raise
        value=read(gates)
        return dict(status=supplement_status(op,value,sha(source)),source_sha256=sha(source),
            extra_checks=str(gates),oracle_sha256=sha(leaf/'oracle.py'),
            passed=value.get('passed'),total=value.get('total'))
    def formal(self,op,source,out,label,leaf=None,kind='CANN_native_reference'):
        leaf=leaf or self.batch/self.q.records[op]['catalog_leaf']
        from catalog_snapshot import freeze_catalog
        root=leaf.parents[1];snapshot=self.batch/'catalog-snapshots'/('formal-'+label+'-'+op)
        records=[dict(catalog_leaf='catalog/ops/'+f.name,operator=f.name,contract_sha256={n:sha(f/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']}) for f in (root/'ops').iterdir() if f.is_dir()]
        clean=freeze_catalog(root,snapshot,records)/'ops'/op
        result=out/(label+'-1.result.json')
        self._cmd('formal-'+op+'-'+label,[base.PY,THREE/'evaluate_reference.py','--operator',op,'--source',source,
            '--leaf',clean,'--output',out,'--server',self.q.server,'--label',label,'--kind',kind])
        value=read(result)
        return dict(status=value.get('status'),geo_mean=value.get('geo_mean'),num_passed=value.get('num_passed'),num_workloads=value.get('num_workloads'),
            result=str(result),source=str(source),source_sha256=sha(source),backend=kind,
            latency_ms={x['uuid']:x['latency_ms'] for x in value.get('per_workload',[]) if x.get('phase')=='timing' and x.get('latency_ms')})
    def qualifier(self,op,item):
        out=Path(item['source']).parent/'qualification.json'
        if not out.exists():self._cmd('qualify-'+op,[base.PY,HERE/'qualify.py','--operator',op,'--source',item['source'],
            '--leaf',self.batch/self.q.records[op]['catalog_leaf'],'--output',out],900)
        value=read(out);assert value['source_sha256']==item['source_sha256'];return {k:v for k,v in value.items() if k!='cases'}
    def capture(self,op,source,label,leaf=None):
        leaf=leaf or self.batch/self.q.records[op]['catalog_leaf']
        cases=[json.loads(x) for x in (leaf/'timing.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
        # 最大规格有FP32时优先FP32，保持同一case/seed；不替代全部正式输入。
        selected=[x for x in cases if x['inputs']['spec'].get('dtype')=='float32'] or cases
        case=max(selected,key=lambda x:__import__('math').prod(x['inputs']['spec'].get('shape',[1])))
        out=self.batch/'paths'/op/(label+'-'+sha(source)[:12]);out.mkdir(parents=True,exist_ok=True)
        spec=out/'spec.json';payload=dict(operator=op,leaf=str(leaf),source=str(source),source_sha256=sha(source),case=case,implementation_label=label)
        if spec.exists():assert read(spec)==payload
        else:base.save_json(spec,payload)
        trace=out/'trace.json'
        if not trace.exists():self._cmd('trace-'+op+'-'+label,[base.PY,NEXT/'driver.py','--spec',spec,'--side','candidate','--trace',trace],600)
        parser=load('reference_closure_trace',THREE/'trace_summary.py');value=parser.summarize(trace);base.save_json(out/'summary.json',value)
        return {k:v for k,v in value.items() if k!='device_tasks'}
    def alternate(self,op,reference,kind):
        old=self.batch/self.q.records[op]['catalog_leaf'];leaf=self.batch/'comparison-catalogs'/kind/'ops'/op
        leaf.mkdir(parents=True,exist_ok=True)
        for n in ['definition.json','correctness.jsonl','timing.jsonl']:
            f=leaf/n
            if f.exists():assert f.read_bytes()==(old/n).read_bytes()
            else:f.write_bytes((old/n).read_bytes())
        original=(old/'oracle.py').read_text(encoding='utf8')
        content=original+'\n\n# 独立命名的补充NPU参考，原PyTorch合同保留。\n'+reference.read_text(encoding='utf8')
        f=leaf/'oracle.py'
        if f.exists():assert f.read_text(encoding='utf8')==content
        else:f.write_text(content,encoding='utf8')
        base.save_json(leaf.parent.parent/'manifest.json',dict(api_version='v6.2',name=kind,evaluator='native',layout='per-operator'))
        base.save_json(self.batch/'comparison-provenance'/kind/(op+'.json'),dict(kind=kind,source=str(reference),source_sha256=sha(reference),original_contract_sha256=self.q.records[op]['contract_sha256'],
            original_oracle=str(old/'oracle.py'),new_oracle_sha256=sha(f),scope='参考不同；与原合同分列，不计为原参考无条件通过'))
        return leaf

    def audited_alternate(self,op,reference,kind):
        """完整历史oracle已有相同输入函数时只复制一次，避免重复参考设备声明。"""
        import ast
        old=BATCH/self.q.records[op]['catalog_leaf'];leaf=BATCH/'comparison-catalogs'/kind/'ops'/op
        left=ast.parse((old/'oracle.py').read_text(encoding='utf8'));right=ast.parse(reference.read_text(encoding='utf8'))
        gen=lambda tree:next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='gen_inputs')
        assert ast.dump(gen(left))==ast.dump(gen(right)),'新参考输入构建不同'
        declarations=[node for node in right.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REFERENCE_DEVICE' for t in node.targets)]
        assert len(declarations)==1,'参考设备声明应唯一'
        leaf.mkdir(parents=True,exist_ok=True)
        for n in ['definition.json','correctness.jsonl','timing.jsonl','oracle.py']:
            data=reference.read_bytes() if n=='oracle.py' else (old/n).read_bytes();target=leaf/n
            if target.exists():assert target.read_bytes()==data
            else:target.write_bytes(data)
        base.save_json(leaf.parent.parent/'manifest.json',dict(api_version='v6.2',name=kind,evaluator='native',layout='per-operator'))
        base.save_json(BATCH/'comparison-provenance'/kind/(op+'.json'),dict(kind=kind,source=str(reference),source_sha256=sha(reference),
            gen_inputs_ast_identical=True,original_contract_sha256=self.q.records[op]['contract_sha256'],oracle_sha256=sha(leaf/'oracle.py')))
        return leaf
def main():
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=ROOT/'runs'/BATCH.name/'STOP-AFTER-JOB'
        if marker.exists() and marker.read_text().startswith('Controller handoff at own job boundary:'):
            marker.rename(marker.with_name('STOP-AFTER-JOB.handoff-'+str(time.time_ns())))
        r=Runner(BATCH.name);r.state['status']='running';r.state['nightly_started_at']=now();r.save()
        try:
            r.job('build-cann',lambda:r._cmd('build-cann',[base.PY,HERE/'build.py'],1200))
            if not (BATCH/'cann-build.json').exists():raise RuntimeError('CANN build unavailable')
            r.job('make-references',lambda:r._cmd('make-references',[base.PY,HERE/'make_references.py'],120))
            refs=read(BATCH/'reference-index.json')
            # 第一轮先覆盖所有资格；某项不支持不阻塞其余对象。
            for op,item in refs.items():r.job('qualify:'+op,lambda op=op,item=item:r.qualifier(op,item))
            r.job('build-cann-v2',lambda:r._cmd('build-cann-v2',[base.PY,HERE/'build_v2.py'],1200))
            if (BATCH/'cann-build-v2.json').exists():
                r.job('make-references-v2',lambda:r._cmd('make-references-v2',[base.PY,HERE/'make_references.py','--revision','v2'],120))
                for op,item in read(BATCH/'reference-index-v2.json').items():
                    old=r.state['jobs'].get('qualify:'+op,{}).get('result') or {}
                    if old.get('eligible_full_contract'):continue
                    checked=r.job('qualify-v2:'+op,lambda op=op,item=item:r.qualifier(op,item))
                    if checked and checked.get('eligible_full_contract'):refs[op]=item
            # 保留v1/v2崩溃原件；v3修复连续存储描述与4D格式，并显式适配统计输出。
            if (BATCH/'cann-build-v3.json').exists():
                build=read(BATCH/'cann-build-v3.json')
                assert sha(build['source'])==build['source_sha256'] and sha(build['binary'])==build['binary_sha256']
                r.job('make-references-v3',lambda:r._cmd('make-references-v3',[base.PY,HERE/'make_references.py','--revision','v3'],120))
                for op,item in read(BATCH/'reference-index-v3.json').items():
                    checked=r.job('qualify-v3:'+op,lambda op=op,item=item:r.qualifier(op,item))
                    if checked and checked.get('eligible_full_contract'):refs[op]=item
            for op,item in refs.items():
                version='-'+item['revision'] if item.get('revision','v1')!='v1' else ''
                qualification=r.state['jobs'].get('qualify'+version+':'+op,{}).get('result') or {}
                if not qualification.get('eligible_full_contract'):continue
                source=Path(item['source']);out=source.parent
                formal=r.job('cann-formal'+version+':'+op,lambda op=op,source=source,out=out,item=item,version=version:r.formal(op,source,out/'formal','CANN-direct'+version,kind=item['kind']))
                if not formal or formal.get('status')!='PASSED':continue
                gates=out/'extra-checks.json'
                status=r.job('cann-extra'+version+':'+op,lambda op=op,source=source,gates=gates,version=version:r.supplementary(op,source,gates,'cann-extra'+version+'-'+op))
                if status!='passed':continue
                r.job('cann-path'+version+':'+op,lambda op=op,source=source,version=version:r.capture(op,source,'CANN'+version))
                if op in ['fake_quantize_per_tensor_affine_cachemask_out','batch_norm_with_update_functional']:
                    leaf=r.alternate(op,source,'CANN-direct-qualified'+version);candidate=r.starting_source(op)
                    r.job('candidate-vs-cann:'+op,lambda op=op,candidate=candidate,leaf=leaf:r.formal(op,candidate,BATCH/'comparisons'/op/'CANN-direct','HG-KDA-vs-CANN',leaf=leaf,kind='existing_candidate_vs_CANN'))
                    r.job('candidate-extra-vs-cann:'+op,lambda op=op,candidate=candidate,leaf=leaf:r.supplemental_at(op,candidate,
                        BATCH/'comparisons'/op/'CANN-direct/extra-checks.json','candidate-extra-vs-cann-'+op,leaf))
                    r.job('candidate-path:'+op,lambda op=op,candidate=candidate:r.capture(op,candidate,'HG-KDA'))
            # 已审计NPU分解保留独立名称；不能据此重写默认验收。
            parent=PROJECT/'operators/campaigns/acceptance-three-20261026'
            for op in ['fractional_max_pool3d','_conj_physical']:
                ref=parent/'npu-decomposition-catalog-v2/ops'/op/'oracle.py';leaf=r.alternate(op,ref,'NPU-decomposition')
                source=r.starting_source(op)
                r.job('supplemental-full:'+op,lambda op=op,source=source,leaf=leaf:r.formal(op,source,BATCH/'comparisons'/op/'NPU-decomposition','HG-KDA-vs-NPU',leaf=leaf,kind='existing_candidate_vs_named_NPU_decomposition'))
                r.job('supplemental-extra:'+op,lambda op=op,source=source,leaf=leaf:r._cmd('supplemental-extra-'+op,[base.PY,NEXT/'validate.py',
                    '--operator',op,'--source',source,'--leaf',leaf,'--output',BATCH/'comparisons'/op/'NPU-decomposition/extra-checks.json'],1200))
                # 旧job的None只表示命令结束，资格必须检查真实补测JSON。
                gates=BATCH/'comparisons'/op/'NPU-decomposition/extra-checks.json'
                r.job('supplemental-extra-audit:'+op,lambda op=op,source=source,leaf=leaf,gates=gates:r.supplemental_at(
                    op,source,gates,'supplemental-extra-'+op,leaf))
            # 真正未通过者优先：保留奇异输入和默认扫描合同，模型读新参考事实。
            for op in ['linalg_inv_ex','cumulative_trapezoid']:
                analysis=BATCH/'analysis'/op;analysis.mkdir(parents=True,exist_ok=True)
                context=analysis/'agent-context.json'
                if not context.exists():
                    q=r.state['jobs'].get('qualify:'+op,{}).get('result')
                    base.save_json(context,dict(status='current_CANN_reference_audit',operator=op,reference_qualification=q,
                        instructions='原合同与奇异用例不改；不调用PyTorch目标计算替代候选，不伪造info。若存在参考矛盾，保留可复现反例。',
                        previous_inverse_conflict=str(PROJECT/'operators/campaigns/remaining-nine-20261030/analysis/linalg_inv_ex/reference-mathematical-conflict.json')))
                r.state['operators'].setdefault(op,{})
                result=r.job('search:'+op+':1',lambda op=op:r.search(op,1))
                if result:r.consider_best(r.state['operators'][op],result)
                r.save()
            r.wait_idle();r.state.update(status='nightly_completed',finished_at=now(),current_job=None);r.save()
            r.job('delivery-report-v2',lambda:r._cmd('delivery-report-v2',[base.PY,HERE/'report_v2.py'],180))
            r.state['status']='completed';r.state['current_job']=None;r.save()
        except Exception as e:
            r.state.update(status='needs_attention',error=type(e).__name__+': '+str(e));r.save();raise
if __name__=='__main__':main()
