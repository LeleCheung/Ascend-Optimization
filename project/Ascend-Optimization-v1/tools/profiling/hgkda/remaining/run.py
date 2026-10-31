"""九项串行攻关；复用既有正式评测和语义补测，不修改共享服务。"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
THREEWAY = HERE.parent/'threeway'
sys.path.insert(0, str(THREEWAY))
spec = importlib.util.spec_from_file_location('threeway_runner', THREEWAY/'run.py')
prior = importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
base = prior.base


class Runner(prior.Runner):
    def evaluate(self,op,source,out,label):
        manifest=source.parent/'proposal.json';withdrawn=self.batch/'withdrawn-proposals.json'
        if manifest.exists() and withdrawn.exists():
            proposal=base.load(manifest);key=op+':'+proposal['id'];reasons=base.load(withdrawn)
            if key in reasons:
                return dict(status='withdrawn_before_submission',source=str(source),
                    source_sha256=proposal['source_sha256'],reason=reasons[key])
        if manifest.exists():
            proposal=base.load(manifest)
            if proposal.get('native_manifest'):
                safety=source.parent/'assessment/native-preflight.json'
                if not safety.exists():
                    try:
                        self.q.run('native-preflight-'+op+'-'+proposal['id'],base.quote([
                            base.PY,HERE/'check_native.py','--operator',op,'--source',source,
                            '--leaf',self.batch/self.q.records[op]['catalog_leaf'],'--output',safety]),600)
                    except RuntimeError:
                        if not safety.exists():raise
                checked=base.load(safety)
                if checked['source_sha256']!=proposal['source_sha256']:raise self.module.Attention('原生前置补测SHA不同')
                if checked['status']!='passed':
                    return dict(status='native_preflight_failed',source=str(source),
                        source_sha256=proposal['source_sha256'],native_safety=str(safety))
        return super().evaluate(op,source,out,label)

    def process_proposals(self):
        prior.process(self)
        for op,row in self.state.get('operators',{}).items():
            best=row.get('best') or {}
            if not best.get('native_manifest'):continue
            sha=best['source_sha256'];out=self.batch/'source-review'/op/sha[:12]/'native-safety.json'
            def check(op=op,best=best,out=out):
                preflight=Path(best['source']).parent/'assessment/native-preflight.json'
                if preflight.exists():
                    value=base.load(preflight)
                    if value['source_sha256']!=best['source_sha256']:raise self.module.Attention('原生预检身份不同')
                    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(preflight.read_bytes());return value
                try:
                    self.q.run('native-safety-'+op,base.quote([base.PY,HERE/'check_native.py',
                        '--operator',op,'--source',best['source'],'--leaf',self.batch/self.q.records[op]['catalog_leaf'],
                        '--output',out]),600)
                except RuntimeError:
                    if not out.exists():raise
                return base.load(out)
            value=self.job('native-safety:'+op+':'+sha[:12],check)
            if not value or value['source_sha256']!=sha:raise self.module.Attention('原生路径补测身份未知')
            best['native_safety']=str(out)
            if value['status']!='passed':
                path=self.batch/'disqualified-sources.json';denied=base.load(path) if path.exists() else {}
                denied[sha]=dict(operator=op,reason='对齐原生路径追加语义检查失败',evidence=str(out))
                base.save_json(path,denied);self.reconcile_selection()
                row['status']='原生路径补测失败；保留其他合格源码'
                records=self.batch/'knowledge/records.json'
                if records.exists():
                    from knowledge import organize
                    knowledge=base.load(records)
                    for item in knowledge:
                        if item.get('evidence',{}).get('source_sha256')==sha:
                            item['status']='observed_failure'
                            item['summary']='原完整合同与旧补测通过，但新增对齐原生路径补测失败；保留为禁止直接复用的失败约束'
                            item['evidence'].setdefault('supplementary',[]).append(dict(path=str(out),sha256=__import__('hashlib').sha256(out.read_bytes()).hexdigest()))
                    base.save_json(records,knowledge);base.save_json(records.parent/'index.json',organize(knowledge))
            self.save()

    def search(self, op, attempt=1):
        if op == 'expm1_':
            # 先做真实成对采集；这里不默认进入长模拟器，原控制器可复用已有采集。
            analysis=self.batch/'analysis'/op
            spec=analysis/'spec.json'
            if not spec.exists():
                import hashlib
                leaf=self.batch/self.q.records[op]['catalog_leaf']
                case=json.loads((leaf/'timing.jsonl').read_text().splitlines()[0])
                source=self.starting_source(op)
                base.save_json(spec,dict(operator=op,leaf=str(leaf),source=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),case=case,
                    scope='本批起点代表输入的真实成对采集；不冒充新候选'))
            capture=analysis/'paired'
            if not capture.exists():
                self.q.run('paired-no-default-simulator-'+op,base.quote([
                    base.PY,base.HERE/'collect.py','--spec',spec,'--output',capture]),2400)
        return super().search(op,attempt)

    def save(self):
        base.save_json(self.path, self.state)
        lines = ['# 剩余九项攻关进展', '', '固定原合同；下列分数为本批完整调用相对PyTorch的加速比。', '',
                 '|算子|起点复验|当前最佳|完整评测|状态|', '|---|---:|---:|---|---|']
        for op in self.q.records:
            row = self.state.get('operators', {}).get(op, {})
            start = row.get('starting_point') or {};best = row.get('best') or {}
            fmt = lambda r: ('%.3f×'%r['geo_mean']) if isinstance(r.get('geo_mean'), (int, float)) else r.get('status', '—')
            lines.append('|'+op+'|'+fmt(start)+'|'+fmt(best)+'|'+str(best.get('num_passed', '—'))+'/'+str(best.get('num_workloads', '—'))+'|'+row.get('status', '待运行')+'|')
        (self.batch/'进展.md').write_text('\n'.join(lines)+'\n', encoding='utf8')

    def verify(self, op):
        super().verify(op)
        result = self.state['jobs'].get('verify-current:'+op, {}).get('result')
        self.state['operators'].setdefault(op, {})['starting_point'] = result
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', default='remaining-nine-20261030')
    parser.add_argument('--phase', choices=['verify', 'proposals', 'search', 'evidence', 'finalize', 'all'], default='all')
    parser.add_argument('--operators', nargs='*')
    args = parser.parse_args()
    with (base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker=base.ROOT/'runs'/args.batch/'STOP-AFTER-JOB'
        if marker.exists() and marker.read_text().startswith('Controller handoff at own job boundary:'):
            marker.rename(marker.with_name('STOP-AFTER-JOB.handoff-'+str(time.time_ns())))
        r = Runner(args.batch);r.state['active_started_epoch'] = time.time();r.save()
        try:
            ops = args.operators or list(r.q.records)
            if args.phase in {'verify', 'all'}:
                for op in ops:
                    r.verify(op)
            if args.phase in {'proposals', 'all'}:
                r.process_proposals()
            if args.phase in {'search', 'all'}:
                for op in ops:
                    best = r.state['operators'].get(op, {}).get('best') or {}
                    if base.extra_qualification(best) == 'passed' and best.get('geo_mean', 0) > .8:
                        continue
                    attempt = r.state.get('search_recovery_attempts', {}).get(op, 1)
                    result = r.job('search:'+op+':'+str(attempt), lambda op=op, attempt=attempt: r.search(op, attempt))
                    r.consider_best(r.state['operators'].setdefault(op, {}), result);r.save()
                    r.process_proposals()
            if args.phase in {'finalize', 'all'}:
                from finalize_remaining import finalize
                finalize(r)
            if args.phase in {'evidence','all'}:
                evidence=base.module('remaining_evidence',HERE/'evidence.py')
                evidence.collect(r)
            r.wait_idle()
            r.state.update(status=args.phase+'_completed', current_job=None, finished_at=r.module.now());r.save()
        except Exception as error:
            r.state.update(status='needs_attention', error=type(error).__name__+': '+str(error));r.save()
            raise


if __name__ == '__main__':
    main()
