"""宿主tmux可恢复25算子队列；复用KG/KGS和本项目卡锁，一次完整独立复验。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import shlex
import shutil
import sys
import time
from pathlib import Path
from policy import operator_type,reference_documents

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
PROJECT = HERE.parents[3]
ROOT = PROJECT.parents[2]
PY = '/usr/local/python3.11.15/bin/python3.11'


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def quote(args):
    return ' '.join(shlex.quote(str(x)) for x in args)


def extra_qualification(record):
    """已有补测必须满足当前准入；尚未补测的原版仍可进入后续资格验证。"""
    if not record.get('extra_checks'):return 'not_run'
    from validation_status import supplement_status
    try:
        checked=load(Path(record['extra_checks']))
        return supplement_status(checked.get('operator'),checked,record.get('source_sha256'))
    except (OSError,ValueError,KeyError):
        return 'extra_evidence_unavailable'


def retain_unqualified(row, record, qualification):
    entries=row.setdefault('retained_unqualified',[])
    if not any(item.get('source_sha256')==record.get('source_sha256') and
               item.get('result')==record.get('result') and
               item.get('extra_checks')==record.get('extra_checks') and
               item.get('qualification')==qualification for item in entries):
        entries.append(dict(record,qualification=qualification))


class Runner:
    def __init__(self, batch):
        self.module = module('next_queue', TOOLS/'run-nightly.py')
        self.module.save=save_json
        self.q = self.module.Queue(batch, ROOT, 'tle_yy')
        self.batch = self.q.batch
        self.path = self.batch/'next-state.json'
        self.state = load(self.path) if self.path.exists() else dict(status='running', jobs={}, operators={}, started_at=self.module.now(), started_epoch=time.time())
        self.state['status'] = 'running'
        if self.state.get('error'):
            self.state.setdefault('previous_interruptions', []).append(dict(error=self.state.pop('error'), resumed_at=self.module.now()))
        self.recover_local_preparation()
        self.reconcile_selection()
        self.native_method = module('next_native_search', HERE/'native_search.py').native

    def reconcile_selection(self):
        """按本批验收指标重选已完成记录；旧记录和原始评测保持原样。"""
        from validation_status import supplement_status
        from report import verify
        for row in self.state.get('operators',{}).values():
            best=row.get('best') or {};qualification=extra_qualification(best)
            if best and qualification not in {'not_run','passed'}:
                retain_unqualified(row,best,qualification);row.pop('best',None)
        candidates={}
        for job in self.state['jobs'].values():
            result=job.get('result')
            if not isinstance(result,dict) or result.get('status')!='PASSED' or not result.get('source_sha256'):continue
            gates=Path(result['extra_checks']) if result.get('extra_checks') else None
            if not gates or not gates.exists():continue
            check=load(gates);op=check.get('operator')
            if op not in self.q.records or supplement_status(op,check,result['source_sha256'])!='passed':continue
            checked=verify(result)
            if checked.get('status')!='verified' or not checked.get('correctness_passed') or checked.get('extra_status')!='passed':continue
            candidates.setdefault(op,[]).append(dict(result,geo_mean=checked['score']))
        for op,records in candidates.items():
            row=self.state['operators'].setdefault(op,{})
            previous=row.get('best') or {}
            for result in records:self.consider_best(row,result)
            chosen=row.get('best') or {}
            if previous and chosen.get('source_sha256')!=previous.get('source_sha256'):
                self.state.setdefault('selection_events',[]).append(dict(operator=op,
                    previous_source_sha256=previous.get('source_sha256'),previous_speedup=previous.get('geo_mean'),
                    selected_source_sha256=chosen.get('source_sha256'),selected_speedup=chosen.get('geo_mean'),
                    reason='按相同合同的成对加速比重选，原始结果保留',time=self.module.now()))
        self.state['selection_policy']='当前补测通过优先于未补测占位；同合同合格记录按相对PyTorch几何平均加速比择优，绝对延迟保留作辅助证据'

    def recover_local_preparation(self):
        """保留旧记录，只重试已证实未发起设备请求的本地准备失败。"""
        jobs=self.state['jobs']
        for key,job in list(jobs.items()):
            if not key.startswith('baseline:') or not job.get('result'):
                continue
            op=key.split(':',1)[1]
            expected=self.q.records[op]['source_sha256']
            result=job['result']
            if result['source_sha256']==expected:
                continue
            corrected=dict(job)
            corrected['result']=dict(result,attribution='HG-KDA历史起点；旧文件名FlagGems不代表原版实现',legacy_label_corrected=True)
            self.state.setdefault('recovery_events',[]).append(dict(key=key,original=job,reason='历史候选误入原版列，保留原件并更正状态归属',time=self.module.now()))
            jobs['historical-start:'+op]=corrected
            self.state['operators'].setdefault(op,{})['starting_point']=corrected['result']
            self.state['operators'][op].pop('baseline',None)
            del jobs[key]
        for key,job in list(jobs.items()):
            if job.get('status')!='failed' or not job.get('error','').startswith('FileNotFoundError:'):
                continue
            kind,_,op=key.partition(':')
            if kind=='reference-path':
                safe=not any((self.batch/'reference-audits'/op).rglob('*.json')) and not any((self.batch/'reference-specs').glob(op+'-*.json'))
            elif kind=='search':
                workspace=self.q.runtime/'kg-runs'/(op+'-'+self.batch.name+'-attempt-1')
                safe=not workspace.exists() and not (self.batch/'analysis'/op/'spec.json').exists()
            else:
                continue
            if not safe:
                continue
            self.state.setdefault('recovery_events',[]).append(dict(key=key,original=job,reason='缺失父目录且无已提交请求；mkdir修复后重新准备',time=self.module.now()))
            del jobs[key]

    def starting_source(self,op):
        record=self.q.records[op]
        bootstrap=self.batch/'bootstrap.json'
        entries=load(bootstrap) if bootstrap.exists() else {}
        item=entries.get(op)
        if item:
            path=self.batch/item['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise self.module.Attention('历史起点SHA改变')
            return path
        return self.batch/record['starting_source'] if record.get('starting_source') else self.baseline_spec(op)['source']

    def baseline_spec(self,op):
        from baseline import resolve
        return resolve(self.batch,op,self.q.records[op])

    def baseline(self,op):
        spec=self.baseline_spec(op)
        result=self.job(spec['key'],lambda:self.evaluate(op,spec['source'],spec['out'],spec['label']))
        if result and spec.get('adapter'):result['baseline_adapter']=spec['adapter']
        row=self.state['operators'].setdefault(op,{})
        previous=row.get('baseline') or {}
        if previous and result and previous.get('source_sha256')!=result.get('source_sha256'):
            row.setdefault('baseline_history',[]).append(previous)
        return result

    def search_source(self,op,attempt):
        best=self.state['operators'].get(op,{}).get('best') or {}
        qualified=False
        if best.get('extra_checks'):
            from validation_status import supplement_status
            gates=Path(best['extra_checks'])
            qualified=gates.exists() and supplement_status(op,load(gates),best.get('source_sha256'))=='passed'
        if (attempt>1 or qualified) and best.get('status')=='PASSED':
            path=Path(best['source'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=best['source_sha256']:
                raise self.module.Attention('续跑最佳源码身份改变 '+op)
            return path
        return self.starting_source(op)

    def save(self):
        self.module.save(self.path, self.state)
        lines = ['# 新25项夜间实验进展', '', '固定合同，KGS19659完整调用；新源码一次正式复验。队列状态不代表25项已交付。', '', '|算子|FlagGems实测|HG-KDA|完整评测|状态|', '|---|---:|---:|---|---|']
        for target in load(self.batch/'targets.json')['targets']:
            op = target['operator']
            row = self.state['operators'].get(op, {})
            base, best = row.get('baseline') or {}, row.get('best') or {}
            fmt = lambda r: '%.3f×'%r['geo_mean'] if r.get('status')=='PASSED' and isinstance(r.get('geo_mean'), (int,float)) else r.get('status', '—')
            best_text=fmt(best) if best.get('source_sha256')!=base.get('source_sha256') else '—（保留FlagGems）' if best else '—'
            lines.append('|'+op+'|'+fmt(base)+'|'+best_text+'|'+str(best.get('num_passed','—'))+'/'+str(best.get('num_workloads','—'))+'|'+row.get('status', target['status'])+'|')
        (self.batch/'进展.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
        # 已核验报告跟随队列更新；报告异常不影响原始状态和正在进行的设备任务。
        try:
            from report import generate
            generate(self.batch,quiet=True)
        except Exception as error:
            print('REPORT_REQUIRES_REVIEW '+type(error).__name__+': '+str(error),flush=True)

    def wait_idle(self, timeout=900):
        deadline=time.monotonic()+timeout
        last=None
        while True:
            try:
                self.q.idle();return
            except self.module.Attention as error:
                last=str(error)
                if time.monotonic()>=deadline:raise
                time.sleep(10)

    def job(self, key, fn):
        if (self.q.runtime/'STOP-AFTER-JOB').exists():
            raise self.module.Attention('收到任务边界停止标记，保留当前状态；清除标记后可恢复')
        previous = self.state['jobs'].get(key)
        if previous and previous['status'] in {'completed', 'failed'}:
            return previous.get('result')
        # 发生中断时交由函数检查原产物和worker，不创建重复请求。
        self.wait_idle()
        self.state.update(current_job=key)
        self.state['jobs'][key] = dict(status='running', started_at=self.module.now())
        self.save()
        print(json.dumps(dict(event='job_started',job=key,time=self.module.now()),ensure_ascii=False),flush=True)
        try:
            result = fn()
            self.state['jobs'][key].update(status='completed', result=result)
        except self.module.Attention:
            raise
        except Exception as e:
            self.wait_idle()
            result = None
            self.state['jobs'][key].update(status='failed', error=type(e).__name__+': '+str(e))
        self.state['jobs'][key]['finished_at'] = self.module.now()
        self.save()
        print(json.dumps(dict(event='job_finished',job=key,status=self.state['jobs'][key]['status'],
                              error=self.state['jobs'][key].get('error'),time=self.module.now()),ensure_ascii=False),flush=True)
        return result

    def evaluate(self, op, source, out, label):
        result = out/(label+'-1.result.json')
        request = out/(label+'-1.request.json')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if request.exists():
            if not result.exists():
                raise self.module.Attention('已提交请求结果未知，停止自动重发 '+str(request))
            saved_request = load(request)
            if saved_request['implementation']['sources'][0]['content'] != source.read_text(encoding='utf8'):
                raise self.module.Attention('已有请求与源码不同')
        else:
            try:
                self.q.evaluate(op, source, out, label, 1)
            except (RuntimeError, AssertionError):
                if not result.exists():
                    raise
        d = load(result)
        return dict(status=d.get('status'), geo_mean=d.get('geo_mean'), num_passed=d.get('num_passed'), num_workloads=d.get('num_workloads'),
                    source=str(source), source_sha256=digest, result=str(result),
                    latency_ms={r['uuid']:r['latency_ms'] for r in d.get('per_workload', []) if r.get('phase')=='timing' and r.get('latency_ms') is not None})

    @staticmethod
    def consider_best(row, result):
        if not result or result.get('status')!='PASSED' or result.get('num_passed')!=result.get('num_workloads'):
            return
        parent=row.get('best')
        if parent:
            qualification=extra_qualification(parent)
            if qualification not in {'not_run','passed'}:
                retain_unqualified(row,parent,qualification);row.pop('best',None);parent=None
        qualification=extra_qualification(result)
        if qualification not in {'not_run','passed'}:
            retain_unqualified(row,result,qualification);return
        if parent and set(parent.get('latency_ms',{}))!=set(result.get('latency_ms',{})):
            raise RuntimeError('候选性能输入集合不同，不能直接择优')
        def score(record):
            value=record.get('geo_mean')
            return value if isinstance(value,(int,float)) and math.isfinite(value) and value>0 else 0
        # 起点记录没有extra_checks时仅作占位，不能在恢复时挤掉已经完成当前补测的候选。
        # 例如同一log10历史源码的另一条无补测记录，不能绕过已发现的布局失败。
        def rank(record):
            return extra_qualification(record)=='passed',score(record)
        if parent is None or rank(result)>rank(parent):
            row['best']=result

    def supplementary(self, op, source, gates, label):
        """保留真实补测失败；只接受绑定当前源码的检查结果，不吞掉采集故障。"""
        from validation_status import supplement_status
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if not gates.exists():
            try:
                self.q.run(label,quote([PY,HERE/'validate.py','--operator',op,
                    '--leaf',self.batch/self.q.records[op]['catalog_leaf'],
                    '--source',source,'--output',gates]),1200)
            except RuntimeError:
                if not gates.exists():raise
                failed=load(gates)
                if (failed.get('operator')!=op or failed.get('source_sha256')!=digest
                        or failed.get('status')!='failed'):
                    raise
        return supplement_status(op,load(gates),digest)

    def audit_reference(self,op):
        out=self.batch/'reference-audits'/op
        result=out/'result.json'
        if result.exists():return load(result)
        out.mkdir(parents=True,exist_ok=True)
        leaf=self.batch/self.q.records[op]['catalog_leaf']
        source=self.batch/self.q.records[op]['seed']
        cases=[json.loads(line) for line in (leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
        # 先做一个有代表性的真实路径体检；完整性能合同仍在正式KGS评测中覆盖全部输入。
        if len(cases)>1:
            cases=[cases[0], max(cases,key=lambda x: x.get('inputs',{}).get('spec',{}).get('shape',[0])[-1] if isinstance(x.get('inputs',{}).get('spec',{}).get('shape'),list) else 0)]
        rows=[]
        for index,case in enumerate(cases):
            spec=self.batch/'reference-specs'/(op+'-'+str(index)+'.json');case_out=out/('case-'+str(index))
            payload=dict(operator=op,leaf=str(leaf),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),case=case)
            if spec.exists() and load(spec)!=payload:raise self.module.Attention('参考审计输入身份改变')
            if not spec.exists():self.module.save(spec,payload)
            if not (case_out/'result.json').exists():
                if case_out.exists():raise RuntimeError('已有不完整参考采集，保留原件，不自动覆盖')
                self.q.run('reference-path-'+op+'-'+str(index),quote([PY,HERE/'audit_reference.py','--spec',spec,'--output',case_out]),600)
            row=load(case_out/'result.json')
            rows.append(dict(case=case['name'],status=row['status'],eligible=row['eligible_default_comparison'],result=str(case_out/'result.json')))
        eligible=bool(rows) and all(r['eligible'] for r in rows)
        summary=dict(status='npu_paths_observed' if eligible else 'reference_requires_separate_review',eligible_default_comparison=eligible,cases=rows,
                     scope='代表性能输入的真实路径初筛；未覆盖的输入不据此声称无CPU回退，保留完整trace')
        self.module.save(result,summary);return summary

    def search(self, op, attempt=1):
        suffix='attempt-'+str(attempt)
        while len(self.q.plan['passes'])<attempt:
            self.q.plan['passes'].append(dict(self.q.plan['passes'][0]))
        out = self.batch/'results'/op/('search-'+str(attempt))
        out.mkdir(parents=True, exist_ok=True)
        seed=self.search_source(op,attempt)
        inverse_probe=None
        if op=='linalg_inv_ex':
            inverse_probe=self.batch/'analysis'/op/('contract-probe-'+hashlib.sha256(seed.read_bytes()).hexdigest()[:12]+'.json')
            if not inverse_probe.exists():
                self.q.run('inverse-contract-probe',quote([PY,HERE/'probe_inverse.py','--source',seed,
                    '--leaf',self.batch/self.q.records[op]['catalog_leaf'],'--output',inverse_probe]),180)
            probe=load(inverse_probe)
            if probe['source']['sha256']!=hashlib.sha256(seed.read_bytes()).hexdigest():
                raise self.module.Attention('求逆诊断与当前种子源码身份不同')
        if op in {'expm1_', 'native_group_norm'}:
            analysis=self.batch/'analysis'/op
            spec=analysis/'spec.json'
            if not spec.exists():
                leaf=self.batch/self.q.records[op]['catalog_leaf']
                case=json.loads((leaf/'timing.jsonl').read_text().splitlines()[0])
                source=self.starting_source(op)
                self.module.save(spec,dict(operator=op,leaf=str(leaf),source=str(source),
                                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),case=case,scope='新批起点代表输入；不冒充新候选结果'))
            capture=analysis/'paired'
            if not capture.exists():
                self.q.run('paired-'+op,quote([PY,HERE/'collect.py','--spec',spec,'--output',capture,'--simulator']),2400)
            if not (analysis/'agent-context.json').exists():
                self.q.run('context-'+op,quote([PY,HERE/'assemble.py','--capture',capture,'--output',analysis/'agent-context.json']),600)
        overlay = ROOT/'runtime'/self.batch.name/(op+'-search-'+str(attempt))/'sources'
        if not overlay.exists():
            self.q.run('overlay-'+op, quote([PY, TOOLS/'build-kg-overlay.py', '--output', overlay, '--native-context']), 600)
            helper=overlay/'kernelgen/tools/hgkda_debug_budget.py'
            shutil.copyfile(HERE/'debug_budget.py',helper)
            server=overlay/'kernelgen/mcp_server/server.py'
            original=server.read_text(encoding='utf8')
            needle='''    workspace = workspace_from_env()
    try:
        return submit_workspace_debug_job('''
            replacement='''    workspace = workspace_from_env()
    from kernelgen.tools.hgkda_debug_budget import admit
    admission = admit(workspace, request.timeout_seconds)
    if admission['status'] != 'ADMITTED':
        return admission
    request = request.model_copy(update={'timeout_seconds': admission['timeout_seconds']})
    try:
        return submit_workspace_debug_job('''
            if original.count(needle)!=1:raise RuntimeError('诊断预算覆盖层入口不唯一')
            server.write_text(original.replace(needle,replacement),encoding='utf8')
            source = seed
            analysis=self.batch/'analysis'/op
            evidence_path=next((analysis/name for name in ['agent-context-v3.json','agent-context-v2.json','agent-context.json']
                                if (analysis/name).exists()),analysis/'agent-context.json')
            if (analysis/'latest-context.json').exists():
                pointer=load(analysis/'latest-context.json');evidence_path=Path(pointer['path'])
                if hashlib.sha256(evidence_path.read_bytes()).hexdigest()!=pointer['sha256']:
                    raise self.module.Attention('最新上下文身份改变，保留原件待核验')
            from context import bounded_context,host_profile_context
            evidence = bounded_context(load(evidence_path),evidence_path) if evidence_path.exists() else dict(status='新成对分析尚未就绪；使用本轮KGS实际profiling，不能引用其他算子数据')
            evidence['host_profiling']=host_profile_context(analysis,hashlib.sha256(seed.read_bytes()).hexdigest())
            if inverse_probe:
                probe=load(inverse_probe)
                from probe_inverse import summary as inverse_summary
                evidence['inverse_contract_probe']=dict(inverse_summary(probe),path=str(inverse_probe),
                    sha256=hashlib.sha256(inverse_probe.read_bytes()).hexdigest())
            evidence['optimization_seed_identity']=dict(path=str(seed),sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                paired_capture_matches_seed=evidence.get('spec',{}).get('source_sha256')==hashlib.sha256(seed.read_bytes()).hexdigest(),
                scope='成对证据仅代表其spec绑定源码；若续跑种子不同，不能当作新种子的已测结果')
            evidence_snapshot=out/'agent-context-loaded.json'
            refs=TOOLS/'skills/hgkda/references'
            documents=reference_documents(op)
            prior=[];remaining=6000
            for name in documents:
                path=refs/name;body=path.read_text(encoding='utf8')
                if len(body)>remaining:
                    prior.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status='超出上下文限额，按需读取'))
                    continue
                prior.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),text=body,origin='历史经验与失败约束；不是本批实测发现'))
                remaining-=len(body)
            evidence['bounded_historical_prior']=prior
            knowledge_path=self.batch/'knowledge/index.json'
            if knowledge_path.exists():
                from knowledge import retrieve,implementation_examples
                evidence['retrieved_knowledge']=retrieve(load(knowledge_path),operator_type(op),
                    dict(operator=op,compiler='0.6.0+ascend.gitc286cba6'),max_chars=6000)
                evidence['transferable_source_examples']=implementation_examples(load(knowledge_path),operator_type(op),
                    dict(operator=op,compiler='0.6.0+ascend.gitc286cba6'))
            save_json(evidence_snapshot,evidence)
            content = '\n\n# 新25项任务\n本轮目标'+op+'：先修正确性，再用实际输入和profiling改进性能。避免只为用例列表做特化；保留原地/out、NaN/Inf、stride、广播和随机机制。复杂算子检查片上缓冲、归约、同步和布局。每轮记录具体预测与验证。不得调用PyTorch计算目标替代候选；CANN参考只用于对照。先提交第一份可运行的完整KGS候选；首轮正式评测前最多两次诊断debug，每次不超过120秒。不要在尚无完整候选时反复测启动微基准。\n当前证据：\n'+json.dumps(evidence,ensure_ascii=False,indent=2)+'\n'
            baseline = self.state['operators'][op].get('baseline')
            if baseline:
                d = load(Path(baseline['result']))
                summary={k:d.get(k) for k in ['status','geo_mean','num_passed','num_workloads','log']}
                summary['result_file']=baseline['result']
                summary['log']=str(summary.get('log') or '')[:1200]
                cases=[r for r in d.get('per_workload',[]) if r.get('phase')=='timing' or r.get('status')!='PASSED']
                summary['cases']=[{k:(str(r.get(k) or '')[:500] if k=='log' else r.get(k)) for k in ['uuid','phase','status','latency_ms','reference_latency_ms','log']} for r in cases[:16]]
                summary['case_count']=len(cases)
                content += '\n本批起点KGS结果摘要（完整原件见结果路径）：\n'+json.dumps(summary,ensure_ascii=False)+'\n'
            for role in (overlay/'kernelgen/.kernelgen/agents').glob('*.md'):
                if role.stem in {'kernel-coder','kernel-knowledge-coder','kernel-optimize','kernel-profile-analyzer','kernel-knowledge-profile-analyzer'}:
                    base_role=role.read_text(encoding='utf8').split('\n\n# HG-KDA 角色扩展')[0]
                    concise='\n\n# HG-KDA本批分析约束\n结合相同输入的实际执行路径和指标比较，先看绝对值/差值/比率与记录范围，再查源码、IR和热点区域。高活跃比例只是待验证方向。get_profile_context的hgkda_workloads提供准确Native输入，不猜测case规格。优先使用当前结构化证据与作用域兼容知识；历史失败约束按需读取，不将全部历史知识注入上下文。保留原角色工具和正式KGS评测。\n'
                    role.write_text(base_role+concise+content, encoding='utf8')
            self.module.save(out/'input-provenance.json', dict(seed=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                debug_budget=dict(helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),mcp_server_sha256=hashlib.sha256(server.read_bytes()).hexdigest(),max_debug_seconds=120,initial_debug_jobs=2),
                roles={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (overlay/'kernelgen/.kernelgen/agents').glob('*.md')},
                agent_context=str(evidence_path) if evidence_path.exists() else None,
                loaded_context=str(evidence_snapshot),loaded_context_sha256=hashlib.sha256(evidence_snapshot.read_bytes()).hexdigest(),
                attribution='Codex编排/诊断与任务指引；CC/DeepSeek实际生成；原生KGS评测'))
        try:
            workspace = self.native_method(self, op, suffix, seed, overlay, self.q.plan['passes'][0]['budget_seconds'])
        except self.module.Attention:
            budget=self.batch/('budget-'+op+'-'+suffix+'.json')
            if not budget.exists():raise
            workspace=self.q.runtime/'kg-runs'/(op+'-'+self.batch.name+'-'+suffix)
            controller=module('next_stop_worker',HERE/'stop_worker.py')
            stopped=out/'own-worker-stop.json'
            if not stopped.exists():controller.stop(workspace,stopped)
            cleanup=controller.cleanup_debug(workspace,server=self.q.server)
            self.module.save(out/'debug-cleanup.json',cleanup)
            self.wait_idle()
        loaded=[]
        expected_roles=load(out/'input-provenance.json')['roles']
        for role in workspace.rglob('kernel*.md'):
            if not role.is_file() or role.name not in expected_roles:continue
            if role.parent.name!='agents' or role.parent.parent.name!='.kernelgen':continue
            raw=role.read_bytes()
            if 'HG-KDA本批分析约束' not in raw.decode('utf8',errors='replace'):continue
            digest=hashlib.sha256(raw).hexdigest()
            if digest!=expected_roles[role.name]:raise self.module.Attention('物化角色与覆盖层身份不同')
            target=out/'loaded-roles'/(digest+'.md');target.parent.mkdir(exist_ok=True);target.write_bytes(raw)
            loaded.append(dict(role=role.name,path=str(role),sha256=digest,archived=str(target)))
        for entry in loaded:
            canonical=Path(entry['path'])
            runtime=canonical.parents[2]/'.claude/agents'/canonical.name
            if runtime.exists():
                body=lambda path:path.read_text(encoding='utf8').split('\n---\n',1)[-1]
                if body(runtime)!=body(canonical):raise self.module.Attention('Claude角色正文与KG角色不同，保留现场待核验')
                raw=runtime.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                target=out/'loaded-roles'/(digest+'.md');target.write_bytes(raw)
                entry['claude_runtime']=dict(path=str(runtime),sha256=digest,archived=str(target),body_matches=True,transformation='KG将capabilities/mcp_tools等头部转为Claude工具配置；正文相同')
        self.module.save(out/'role-load-evidence.json',dict(roles=loaded,scope='实际工作区已物化角色；是否调用具体分析工具另查原生记录'))
        if not loaded:raise RuntimeError('未找到实际物化的HG-KDA角色，保留原始工作区，不声称已接入')
        archive = out/'native-evidence'
        if not archive.exists():
            self.q.run('archive-'+op, quote([PY, PROJECT/'tools/evaluation/archive-kg-evidence.py', workspace, archive]), 900)
        candidates = []
        seen = set()
        from selection import search_rank
        for directory in workspace.glob('stages/optimize/work/.kernelgen/evals/round-*'):
            result = directory/'result.json'
            src = directory/'main.py'
            if not result.exists() or not src.exists():
                continue
            d = load(result)
            if d.get('status')!='PASSED' or d.get('is_hack') or d.get('num_passed')!=d.get('num_workloads'):
                continue
            solution = load(directory/'solution.json').get('sources', [])
            if len(solution)!=1 or solution[0].get('content')!=src.read_text(encoding='utf8'):
                continue
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rank = search_rank(d)
            if rank is not None:
                candidates.append((rank, src))
        if not candidates:
            return dict(status='no_valid_candidate', workspace=str(workspace))
        _, src = min(candidates, key=lambda p:p[0])
        frozen = out/'selected.py'
        if frozen.exists() and frozen.read_bytes()!=src.read_bytes():
            raise self.module.Attention('恢复的最佳源码发生变化')
        frozen.write_bytes(src.read_bytes())
        result = self.evaluate(op, frozen, out/'independent', 'HG-KDA')
        if result['status']=='PASSED' and result['num_passed']==result['num_workloads']:
            gates = out/'extra-checks.json'
            qualification=self.supplementary(op,frozen,gates,'extra-'+op)
            result['extra_checks'] = str(gates)
            result['supplementary_status']=qualification
            if qualification!='passed':
                result['status'] = 'supplement_failed'
            else:
                # 自动固化已测量实现与作用域，不把模型猜测写成已证实的优化因果。
                from knowledge import organize
                formal=Path(result['result']);request=formal.with_name(formal.name.replace('.result.json','.request.json'))
                record=dict(id=op+'-'+result['source_sha256'][:12],operator_type=operator_type(op),
                    action_id='validated-implementation-'+op,scope=dict(operator=op,compiler='0.6.0+ascend.gitc286cba6'),
                    summary='本批完整KGS与补充合同通过的实现；源码和实测成绩可复用，具体瓶颈因果仍须查分析证据',
                    status='validated',updated_at=self.module.now(),
                    evidence=dict(source=str(frozen),source_sha256=result['source_sha256'],formal_result=str(formal),
                        formal_result_sha256=hashlib.sha256(formal.read_bytes()).hexdigest(),formal_request=str(request),
                        supplementary=[dict(path=str(gates),sha256=hashlib.sha256(gates.read_bytes()).hexdigest())]))
                knowledge=self.batch/'knowledge';knowledge.mkdir(exist_ok=True)
                records_path=knowledge/'records.json';records=load(records_path) if records_path.exists() else []
                records.append(record);self.module.save(records_path,records);self.module.save(knowledge/'index.json',organize(records))
        return result

    def local_catalog_cache_failure(self, op):
        """仅对已结束、尚无评测的字节码目录拒绝允许新attempt；不重提未知请求。"""
        workspace=self.q.runtime/'kg-runs'/(op+'-'+self.batch.name+'-attempt-1')
        metadata=workspace/'.kernelgen/run-process.json';progress=workspace/'.kernelgen/run-progress.json'
        if not metadata.exists() or not progress.exists():return False
        process=load(metadata);status=load(progress)
        if not process.get('finished_at') or process.get('state')!='EXITED':return False
        message=status.get('message','')
        if status.get('state')!='INFRASTRUCTURE_ERROR' or 'unexpected files' not in message or '__pycache__' not in message:
            return False
        active=workspace/'.kernelgen/active-server-operations.json'
        if active.exists() and load(active).get('operations'):return False
        if (workspace/'stages/optimize').exists():return False
        return True

    def execute(self):
        # 等锁的时间不消耗实验预算；重新启动也沿用本批首次真正执行时间。
        if not self.state.get('active_started_epoch'):
            self.state['active_started_epoch']=time.time()
            self.save()
        version=self.batch/'baseline-verified.json'
        if not version.exists():
            self.job('verify-software',lambda:self.q.run('verify-software',quote([PY,TOOLS/'verify-baseline.py','--server',self.q.server,'--output',version]),600))
        if not version.exists() or load(version).get('status')!='verified':raise self.module.Attention('固定版本未通过核验')
        from proposals import process as process_proposals
        process_proposals(self)
        priority = self.q.plan['priority']
        order = [op for op in priority if op in self.q.records]+[op for op in self.q.records if op not in priority]
        deadline = self.state['active_started_epoch']+3600*self.q.plan['duration_hours']
        # 先补齐未知原版，避免少数难算子的长搜索占满整夜而其余目标仍没有基线。
        for op in order:
            if self.state['jobs'].get(self.baseline_spec(op)['key'],{}).get('status') in {'completed','failed'}:
                continue
            if time.time()>deadline or (self.q.runtime/'STOP-AFTER-JOB').exists():
                self.state['status']='budget_boundary';self.save();return
            process_proposals(self)
            row=self.state['operators'].setdefault(op,{})
            row['status']='原版完整评测';self.save()
            row['baseline']=self.baseline(op)
            self.consider_best(row,row.get('baseline'))
            row['status']='原版合同通过，待补测与路径核验' if (row.get('baseline') or {}).get('status')=='PASSED' else '原版未通过，待修复与优化'
            self.save()
        for op in order:
            if time.time()>deadline or (self.q.runtime/'STOP-AFTER-JOB').exists():
                self.state['status']='budget_boundary'
                self.save()
                return
            process_proposals(self)
            row = self.state['operators'].setdefault(op, {})
            row['status']='起点评测';self.save()
            source=self.baseline_spec(op)['source']
            row['baseline'] = self.baseline(op)
            start=self.starting_source(op)
            if start.read_bytes()!=source.read_bytes():
                row['starting_point']=self.job('historical-start:'+op,lambda:self.evaluate(op,start,self.batch/'results'/op/'historical-start','HG-KDA-start'))
                self.consider_best(row,row['starting_point'])
            audit=self.job('reference-path:'+op,lambda:self.audit_reference(op))
            row['reference_path_status']=(audit or {}).get('status','unknown')
            self.consider_best(row,row.get('baseline'))
            best=row.get('best') or {}
            if best.get('status')=='PASSED' and best.get('geo_mean',0)>.8 and not best.get('extra_checks'):
                # 原版或历史实现复测已过线时先补测，再决定是否需要长搜索。
                gates=(self.baseline_spec(op)['out'] if best.get('source_sha256')==self.baseline_spec(op)['source_sha256'] else Path(best['result']).parent)/'extra-checks.json'
                if not gates.exists():
                    self.job('qualify-best:'+op+':'+best['source_sha256'][:12],lambda:self.q.run('qualify-best-'+op,
                        quote([PY,HERE/'validate.py','--operator',op,'--leaf',self.batch/self.q.records[op]['catalog_leaf'],
                               '--source',best['source'],'--output',gates]),1200))
                if gates.exists():
                    from validation_status import supplement_status
                    best['extra_checks']=str(gates)
                    if supplement_status(op,load(gates),best['source_sha256'])!='passed':
                        row.setdefault('retained_unqualified',[]).append(dict(best,qualification='baseline_supplement_failed'))
                        row.pop('best',None);best={}
                self.save()
            if best.get('status')=='PASSED' and best.get('geo_mean',0)>.8 and best.get('extra_checks'):
                gates=load(Path(best['extra_checks']))
                from validation_status import supplement_status
                if supplement_status(op,gates,best.get('source_sha256'))=='passed':
                    row['status']='合同及补充检查通过，超过0.8×' if (audit or {}).get('eligible_default_comparison') else '数值和性能通过，参考路径需单列核验'
                    self.save();continue
            row['status']='KG/CC优化';self.save()
            best = self.job('search:'+op, lambda:self.search(op))
            if best is None and self.local_catalog_cache_failure(op):
                best=self.job('search-retry:'+op+':2',lambda:self.search(op,attempt=2))
            if best and best.get('status')=='PASSED':
                self.consider_best(row,best)
                row['status']='合同通过且超过0.8×' if row['best'].get('geo_mean',0)>.8 else '合同通过，性能未达标'
                if not (audit or {}).get('eligible_default_comparison'):
                    row['status']='完整合同通过，参考路径需单列核验'
            elif row.get('best'):
                row['status']='保留有效起点；搜索未交付'
            else:
                row['status']='本轮未交付；失败原件保留'
            self.save()
        self.state.update(status='pass_completed', current_job=None, finished_at=self.module.now())
        self.save()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',default='next25-20261029');a=p.parse_args()
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=Runner(a.batch)
        try:r.execute()
        except Exception as e:
            r.state.update(status='needs_attention',error=type(e).__name__+': '+str(e));r.save();raise
