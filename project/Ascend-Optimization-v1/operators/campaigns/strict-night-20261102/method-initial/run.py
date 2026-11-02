"""严格Triton持续优化队列：固定CANN对照，持卡锁，已知终态才交接。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
PROJECT = HERE.parents[3]
ROOT = PROJECT.parents[2]
OLD = PROJECT/'operators/campaigns/strict-triton-20261102'
BATCH = PROJECT/'operators/campaigns/strict-night-20261102'
PY = '/usr/local/python3.11.15/bin/python3.11'
PRIORITY = ['hardswish', 'fix', 'hardshrink_backward', 'rad2deg_', 'round_out',
            'special_round_out', 'hardswish_out', 'log2_', 'fmod_tensor_']
REMAINDER = ['cumulative_trapezoid', 'linalg_inv_ex', 'expm1_', 'uniform', 'fractional_max_pool3d']


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf8'); tmp.replace(p)
def module(name, p):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def quote(args): return ' '.join(shlex.quote(str(v)) for v in args)


def prepare():
    if (BATCH/'plan.json').exists(): return
    BATCH.mkdir(parents=True, exist_ok=True)
    previous = read(OLD/'delivery.json')
    assert previous['accepted'] == 16
    save(BATCH/'starting-delivery.json', previous)
    plan = read(OLD/'plan.json')
    rows = {r['operator']: r for r in previous['rows']}
    for record in plan['records']:
        op = record['operator']; row = rows[op]
        seed = BATCH/'starting-points'/(op+'.py'); seed.parent.mkdir(exist_ok=True)
        source = Path(row['source']); assert sha(source) == row['source_sha256']
        shutil.copyfile(source, seed)
        for name, origin in [('original-catalog', OLD/'catalog/ops'/op),
                             ('catalog', OLD/('cann-catalog' if row.get('cann') else 'catalog')/'ops'/op)]:
            dest = BATCH/name/'ops'/op
            shutil.copytree(origin, dest, ignore=shutil.ignore_patterns('__pycache__'))
        record.update(seed=str(seed.relative_to(BATCH)), source_sha256=sha(seed),
                      catalog_leaf='catalog/ops/'+op,
                      contract_sha256={n:sha(BATCH/'catalog/ops'/op/n)
                                       for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']},
                      reference='qualified_CANN' if row.get('cann') else 'original_PyTorch_only')
    for name in ['catalog', 'original-catalog']:
        save(BATCH/name/'manifest.json', dict(api_version='v6.2', name=BATCH.name+'-'+name,
                                            evaluator='native', layout='per-operator'))
    plan.update(schema='hgkda-strict-night-v1', priority=PRIORITY, remainder=REMAINDER,
                duration_hours=18, passes=[dict(rounds=4,budget_seconds=2400)],
                starting_delivery_sha256=sha(OLD/'delivery.json'),
                baseline_commit='a4cb1b30f9b93d600a0c50d698af1f14d826bb39',
                candidate_backend='strict_Triton', reference_policy='合格原生CANN；缺口仅按原语义搜索，不计CANN达标',
                measurement_policy='每份新源码默认一次完整评测；原合同不变；禁止同SHA重复挑分')
    save(BATCH/'plan.json', plan)


class Runner:
    def __init__(self):
        self.module = module('night_queue', TOOLS/'run-nightly.py')
        self.module.save = save
        self.q = self.module.Queue(BATCH.name, ROOT, 'tle_yy')
        self.batch = BATCH
        self.path = BATCH/'state.json'
        self.state = read(self.path) if self.path.exists() else dict(status='running', jobs={}, operators={}, started_at=self.module.now())
        self.previous = {r['operator']: r for r in read(BATCH/'starting-delivery.json')['rows']}
        sys.path.insert(0, str(TOOLS/'next_phase'))
        self.native_method = module('night_native', TOOLS/'next_phase/native_search.py').native
    def persist(self): save(self.path, self.state); self.report()
    def wait(self):
        end = time.monotonic()+900
        while True:
            try: self.q.idle(); return
            except self.module.Attention:
                if time.monotonic()>end: raise
                time.sleep(10)
    def cmd(self,key,args,budget=2100):
        self.wait(); return self.q.run(key, quote(args), budget)
    def context(self,op,seed):
        old = self.previous[op]
        evidence = dict(operator=op, seed=dict(path=str(seed),sha256=sha(seed)),
                        starting_CANN=old.get('cann'), starting_PyTorch=old.get('pytorch'),
                        device_path=old.get('device_path'), compiled_path=old.get('compiled_path'))
        formal = old.get('cann') or old.get('pytorch')
        if formal:
            v = read(formal['path'])
            evidence['per_case'] = [{k:r.get(k) for k in ['uuid','phase','status','latency_ms','reference_latency_ms']}
                                    for r in v.get('per_workload',[]) if r.get('phase')=='timing']
        evidence['diagnosis'] = (
            '固定真实CANN参考，目标完整调用几何平均>0.8。检查大输入program数量、单program流式循环与访存组织；'
            '区分宿主发射与设备耗时。分块需控制UB，不得只调小输入延迟。对照源码/实际IR/trace按需读取；'
            '不得把待验证解释写成结论。允许编译缓存和当前stream getter，禁止缓存输出或旧stream。'
            '源码含_strict存储工具时仍须保留非连续分支。先形成一份真实可运行改进并完整evaluate；'
            '首轮前最多两次debug，每次<=120秒。支持普通shape/dtype策略，不按用例名/seed特化。'
            '候选必须完全自包含的Python+Triton；不允许外部.so、CANN/AscendC/PyTorch计算/原生DMA。'
            '当前适合先试减少program重复调度或增大片内顺序处理；用profile验证，别只重测同源码。')
        if op in REMAINDER:
            evidence['unresolved'] = old['status']
            extra = old.get('supplementary') or {}
            if extra.get('path'): evidence['supplementary_failures'] = read(extra['path'])
            evidence['diagnosis'] += ('这项参考或数值仍存在缺口，先严格修复原语义，不能通过修改容差或参考规避。'
                                     'uniform禁止重写原生CANN参考以掩盖下界舍入问题；fractional没有原生接口不能伪造。')
        return evidence
    def overlay(self,op,attempt,seed):
        output = ROOT/'runtime'/BATCH.name/(op+'-search-'+str(attempt))/'sources'
        if output.exists():
            assert (output/'strict-night-overlay.json').exists(), '覆盖层准备未完成'
            return output
        self.cmd('overlay-'+op+'-'+str(attempt),[PY,TOOLS/'build-kg-overlay.py','--output',output,'--native-context'],600)
        shutil.copyfile(HERE/'admission.py',output/'kernelgen/tools/hgkda_strict_admission.py')
        adapter = output/'kernelgen/tools/kernelgen_server_adapter.py'
        text = adapter.read_text()
        for name in ['preflight_bundle','evaluate_bundle']:
            start=text.index('def '+name+'(')
            i=text.index('    from kernelgen_client.http import ',start)
            text=text[:i]+('    from kernelgen.tools.hgkda_strict_admission import admit_bundle\n'
                            '    admit_bundle(bundle, context)\n\n')+text[i:]
        adapter.write_text(text)
        # 复用已有有界诊断；只改本批overlay。
        shutil.copyfile(TOOLS/'next_phase/debug_budget.py', output/'kernelgen/tools/hgkda_debug_budget.py')
        server = output/'kernelgen/mcp_server/server.py'; text=server.read_text()
        needle='    workspace = workspace_from_env()\n    try:\n        return submit_workspace_debug_job('
        replacement=('    workspace = workspace_from_env()\n'
                     '    from kernelgen.tools.hgkda_debug_budget import admit\n'
                     '    admission = admit(workspace, request.timeout_seconds)\n'
                     "    if admission['status'] != 'ADMITTED':\n        return admission\n"
                     "    request = request.model_copy(update={'timeout_seconds': admission['timeout_seconds']})\n"
                     '    try:\n        return submit_workspace_debug_job(')
        assert text.count(needle)==1;server.write_text(text.replace(needle,replacement))
        evidence = self.context(op,seed)
        save(BATCH/'analysis'/op/('search-'+str(attempt)+'.json'),evidence)
        content='\n\n# HG-KDA严格Triton夜间任务\n'+json.dumps(evidence,ensure_ascii=False,indent=2)+'\n'
        roles={}
        for role in (output/'kernelgen/.kernelgen/agents').glob('*.md'):
            if role.stem in {'kernel-coder','kernel-knowledge-coder','kernel-optimize','kernel-profile-analyzer','kernel-knowledge-profile-analyzer'}:
                # 用本批精简上下文代替全部历史混合后端模板，保留原角色和严格规则。
                base=role.read_text().split('\n\n# HG-KDA 角色扩展')[0]
                role.write_text(base+content)
            roles[role.name]=sha(role)
        save(output/'strict-night-overlay.json',dict(operator=op,attempt=attempt,roles=roles,
                 admission_sha256=sha(output/'kernelgen/tools/hgkda_strict_admission.py'),
                 adapter_sha256=sha(adapter),seed_sha256=sha(seed)))
        save(BATCH/'analysis'/op/('overlay-'+str(attempt)+'.json'),read(output/'strict-night-overlay.json'))
        return output
    def search(self,op,attempt):
        while len(self.q.plan['passes'])<attempt:
            self.q.plan['passes'].append(dict(self.q.plan['passes'][0]))
        row=self.state['operators'].setdefault(op,dict(attempts=[]))
        key=op+':search-'+str(attempt)
        if self.state['jobs'].get(key,{}).get('status')=='completed':return
        seed=Path((row.get('best') or {}).get('source', BATCH/self.q.records[op]['seed']))
        self.state.update(status='running',current_job=key)
        self.state['jobs'][key]=dict(status='running',seed=str(seed),source_sha256=sha(seed));self.persist()
        overlay=self.overlay(op,attempt,seed)
        # native_search的exports不覆盖这三个变量，docker exec显式传入。
        # 变量由独立run-kg脚本导出，宿主环境不泄露任何模型凭据。
        wrapper=ROOT/'runtime'/BATCH.name/'run-kg-strict.sh'
        if not wrapper.exists():
            script=(PROJECT/'tools/evaluation/run-kg-operator-910b.sh').read_text()
            # 原脚本通过自身位置定位repo；复制版必须显式使用固定repo。
            script=script.replace('repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)',
                                  'repo='+shlex.quote(str(self.q.repo)))
            script=script.replace('"$(dirname "$0")/wait-kg-worker.py"',
                                  shlex.quote(str(PROJECT/'tools/evaluation/wait-kg-worker.py')))
            script=script.replace('set -euo pipefail','set -euo pipefail\nexport HGKDA_STRICT_BATCH='+shlex.quote(str(BATCH))+
                 '\nexport HGKDA_RUNTIME_AUDIT='+shlex.quote(str(TOOLS/'strict_triton/runtime_audit.py')))
            wrapper.parent.mkdir(parents=True,exist_ok=True);wrapper.write_text(script)
        # 只替换native helper源码中的启动脚本路径，保留它的请求身份/超时保护。
        helper=ROOT/'runtime'/BATCH.name/'native_search.py'
        if not helper.exists():
            text=(TOOLS/'next_phase/native_search.py').read_text().replace(
                "self.q.project/'tools/evaluation/run-kg-operator-910b.sh'",repr(str(wrapper)))
            helper.write_text(text)
        method=module('strict_night_native_'+str(attempt),helper).native
        self.wait()
        try:
            workspace=method(self,op,'attempt-'+str(attempt),seed,overlay,2400)
        except self.module.Attention:
            if not (BATCH/('budget-'+op+'-attempt-'+str(attempt)+'.json')).exists():raise
            workspace=self.q.runtime/'kg-runs'/(op+'-'+BATCH.name+'-attempt-'+str(attempt))
            stop=module('night_stop',TOOLS/'next_phase/stop_worker.py')
            stop.stop(workspace,BATCH/'analysis'/op/('own-worker-stop-'+str(attempt)+'.json'))
            save(BATCH/'analysis'/op/('cleanup-'+str(attempt)+'.json'),stop.cleanup_debug(workspace))
            self.wait()
        result=self.collect(op,attempt,workspace)
        row['attempts'].append(result)
        self.state['jobs'][key].update(status='completed',result=result,finished_at=self.module.now());self.persist()
    def collect(self,op,attempt,workspace):
        candidates=[];record=dict(workspace=str(workspace),attempt=attempt,candidates=[])
        old_sha=self.previous[op]['source_sha256']
        for result in sorted(workspace.glob('stages/optimize/work/.kernelgen/evals/round-*/result.json')):
            source=result.with_name('main.py'); v=read(result)
            if not source.exists() or v.get('status')!='PASSED' or v.get('is_hack') or sha(source)==old_sha:continue
            if v.get('num_passed')!=v.get('num_workloads') or not v.get('num_workloads'):continue
            solution=read(result.with_name('solution.json')).get('sources',[])
            if len(solution)!=1 or solution[0].get('content')!=source.read_text(encoding='utf8'):
                raise self.module.Attention('KG记录与实际源码不一致')
            candidates.append((float(v.get('geo_mean') or 0),source,result))
        # 不重测KG已经完整评测过的同一源码；按CANN结果排序后补验原语义。
        seen=set()
        for score,source,result in sorted(candidates,reverse=True):
            if sha(source) in seen:continue
            seen.add(sha(source))
            out=BATCH/'candidates'/op/(sha(source)[:16]);out.mkdir(parents=True,exist_ok=True)
            target=out/'main.py'
            if target.exists():assert sha(target)==sha(source)
            else:shutil.copyfile(source,target)
            shutil.copyfile(result,out/'search-result.json')
            entry=dict(source=str(target),source_sha256=sha(target),search_result=str(result),search_score=score,
                       provenance='KG/HG-KDA与CC/DeepSeek生成；Codex诊断、编排与门禁',
                       reference=self.q.records[op]['reference'])
            # 至多验证本次前三个不同源码；未选项保留KG完整结果。
            if len(record['candidates'])>=3:break
            entry.update(self.validate(op,target,out))
            record['candidates'].append(entry)
            row=self.state['operators'][op]
            baseline=(self.previous[op].get('cann') or self.previous[op].get('pytorch') or {}).get('score',0)
            best_score=(row.get('best') or {}).get('search_score',baseline)
            if entry['qualification']=='passed' and score>best_score:
                row['best']=entry
        record['status']='candidates_evaluated' if record['candidates'] else 'no_new_valid_candidate'
        return record
    def validate(self,op,source,out):
        original=BATCH/'original-catalog/ops'/op
        from validation_status import supplement_status
        extras={}
        for label,leaf in [('pytorch',original),('cann',BATCH/'catalog/ops'/op)]:
            if label=='cann' and self.q.records[op]['reference']!='qualified_CANN':continue
            output=out/('extra-'+label+'.json')
            if not output.exists():
                try:self.cmd('extra-'+op+'-'+sha(source)[:8]+'-'+label,
                    [PY,TOOLS/'cann_priority/validate_cann.py','--operator',op,'--source',source,'--leaf',leaf,'--output',output],1200)
                except RuntimeError:
                    if not output.exists():raise
            extras[label]=dict(path=str(output),status=supplement_status(op,read(output),sha(source)))
        formal=out/'pytorch/strict-PyTorch-1.result.json'
        if not formal.exists():
            self.cmd('pytorch-'+op+'-'+sha(source)[:8],[PY,TOOLS/'cann_priority/evaluate.py','--operator',op,
                '--source',source,'--leaf',original,'--output',out/'pytorch','--server',self.q.server,
                '--label','strict-PyTorch','--kind','strict_Triton_candidate'])
        value=read(formal)
        qualification='passed' if (value.get('status')=='PASSED' and not value.get('is_hack') and
                         value.get('num_passed')==value.get('num_workloads') and
                         all(v['status']=='passed' for v in extras.values())) else 'failed'
        return dict(qualification=qualification,extras=extras,pytorch_result=str(formal),
                    pytorch_score=value.get('geo_mean'),review_status='pending_Codex_manual_review',
                    compiled_status='pending_selected_verification')
    def report(self):
        accepted=sum(r['accepted'] for r in self.previous.values());provisional=[]
        for op,row in self.state.get('operators',{}).items():
            best=row.get('best') or {}
            if best.get('reference')=='qualified_CANN' and best.get('search_score',0)>.8 and not self.previous[op]['accepted']:
                provisional.append(op)
        save(BATCH/'progress.json',dict(status=self.state['status'],current_job=self.state.get('current_job'),
              frozen_accepted=accepted,new_numerically_qualified=provisional,
              scope='新增项还须Codex逐路径审查与实际TTIR/npubin/设备路径核验，未核验前不计正式交付'))
        lines=['# 严格Triton夜间优化进展','',f'冻结起点：{accepted}/30；本轮状态：{self.state["status"]}。',
               '', '| 算子 | 起点相对CANN | 新候选相对CANN | 原语义及补测 | 后端审查 |',
               '| --- | ---: | ---: | --- | --- |']
        for op in PRIORITY+REMAINDER:
            old=self.previous[op];best=(self.state.get('operators',{}).get(op) or {}).get('best') or {}
            a=old.get('cann',{}).get('score');b=best.get('search_score') if best.get('reference')=='qualified_CANN' else None
            lines.append('| '+op+' | '+(f'{a:.4f}×' if a is not None else '参考待解决')+' | '+
                         (f'{b:.4f}×' if b is not None else '待完成')+' | '+best.get('qualification','待完成')+' | '+
                         best.get('review_status','待完成')+' |')
        (BATCH/'夜间优化进展.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    def execute(self):
        self.state.update(status='running');self.persist()
        for attempt in [1,2]:
            for op in PRIORITY+REMAINDER:
                best=(self.state['operators'].get(op) or {}).get('best') or {}
                if attempt>1 and (best.get('search_score',0)>.85 or op in {'uniform','fractional_max_pool3d'}):continue
                self.search(op,attempt)
        self.wait();self.state.update(status='search_completed_pending_review',current_job=None,finished_at=self.module.now());self.persist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare()
    if a.prepare:print(BATCH);return
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        runner=Runner()
        try:runner.execute()
        except Exception as error:
            runner.state.update(status='requires_review',error=type(error).__name__+': '+str(error));runner.persist();raise


if __name__=='__main__':main()
