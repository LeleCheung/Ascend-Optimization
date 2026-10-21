#!/usr/bin/env python3
"""910B宿主的串行夜间队列；持卡锁，保存失败，已知结果可恢复，未知设备请求停止。"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request


def now():
    return datetime.datetime.utcnow().isoformat() + 'Z'


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Attention(RuntimeError):
    pass


class Queue:
    def __init__(self, batch, root, container):
        self.root, self.container = root, container
        self.repo = root / 'Ascend-Optimization'
        self.project = self.repo / 'project/Ascend-Optimization-v1'
        self.batch = self.project / 'operators/campaigns' / batch
        self.plan = json.loads((self.batch / 'plan.json').read_text())
        self.runtime = root / 'runs' / batch
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.path = self.batch / 'queue-state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'batch': batch, 'created_at': now(), 'status': 'created', 'jobs': {}}
        fingerprint = hashlib.sha256((self.batch / 'plan.json').read_bytes()).hexdigest()
        if self.state.get('plan_sha256', fingerprint) != fingerprint:
            raise Attention('恢复时plan已变化，停止混用实验')
        self.state['plan_sha256'] = fingerprint
        self.server = self.plan['server']
        self.py = '/usr/local/python3.11.15/bin/python3.11'
        self.records = {x['operator']: x for x in self.plan['records']}
        for record in self.records.values():
            for key in ['seed', 'catalog_leaf']:
                if '\\' in record[key]:
                    raise Attention('plan包含Windows路径，请生成POSIX合同路径')
            source = self.batch/record['seed']
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=record['source_sha256']:
                raise Attention('原版源码缺失或SHA不符: '+str(source))
            if not (self.batch/record['catalog_leaf']/'definition.json').is_file():
                raise Attention('合同缺失: '+record['operator'])
            for name,digest in record.get('contract_sha256',{}).items():
                if hashlib.sha256((self.batch/record['catalog_leaf']/name).read_bytes()).hexdigest()!=digest:
                    raise Attention('合同已变化: '+record['operator']+'/'+name)
        self.overlay = root / 'runtime' / batch / 'sources'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def idle(self):
        for port in [19655, 19657, 19659]:
            try:
                value = json.load(self.opener.open('http://127.0.0.1:%d/status' % port, timeout=15))['scheduler']
            except Exception as error:
                raise Attention('无法确认服务状态: ' + str(error))
            if value['healthy'] != 1 or any(value[k] for k in ['active', 'waiting', 'broken']):
                raise Attention('服务忙或异常: ' + str(port))

    def run(self, key, body, budget=10800):
        log = self.runtime / (key.replace(':', '-') + '.log')
        command = '''set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONPATH=%s:%s:%s:%s
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
%s
''' % (self.repo/'third_party/FlagGems/src', self.repo/'sources/kernelgen_server/client',
        self.repo/'sources/kernelgen_server', self.repo/'sources', body)
        # 追加日志并记录一次执行边界；恢复不覆盖先前错误。
        with log.open('ab') as stream:
            stream.write(('\nEXECUTION ' + now() + '\n').encode());stream.flush()
            process = subprocess.Popen(['docker', 'exec', '-i', self.container, 'bash', '-s'],
                stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT)
            process.stdin.write(command.encode());process.stdin.close()
            start = time.monotonic()
            while process.poll() is None:
                if time.monotonic() - start > budget:
                    raise Attention('单步骤超过预算；保留进程待检查，不继续提交设备任务: ' + key)
                time.sleep(10)
        if process.returncode:
            raise RuntimeError('步骤失败，见 ' + str(log))

    def evaluate(self, op, source, output, label, repeats):
        # 请求已提交但结果未知时，不盲目重发。
        for i in range(1, repeats+1):
            request = output / (label+'-%d.request.json'%i)
            result = output / (label+'-%d.result.json'%i)
            if request.exists() and not result.exists():
                raise Attention('评测请求结果未知: ' + str(request))
        args = [self.py, str(self.project/'tools/evaluation/evaluate-operator.py'),
            '--operator', op, '--operator-dir', str(self.batch/self.records[op]['catalog_leaf']),
            '--source', str(source), '--label', label, '--output', str(output),
            '--server', self.server, '--repeats', str(repeats), '--timing-scope', 'walltime']
        self.idle()
        self.run('evaluate:'+op+':'+label, ' '.join(shlex.quote(x) for x in args))
        values = [json.loads((output/(label+'-%d.result.json'%i)).read_text()) for i in range(1,repeats+1)]
        assert all(v['status']=='PASSED' and v['num_passed']==v['num_workloads'] for v in values)
        return {'status': 'passed', 'scores': [v.get('geo_mean') for v in values],
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'workloads': values[0]['num_workloads'], 'output': str(output)}

    def baseline(self, op):
        return self.evaluate(op, self.batch/self.records[op]['seed'], self.batch/'results'/op/'FlagGems', 'FlagGems', 1)

    def kg(self, op, arm):
        self.idle()
        suffix = self.batch.name + '-' + arm
        run_root = self.runtime/'kg-runs'
        workspace = run_root/(op+'-'+suffix)
        output = self.batch/'results'/op/arm
        output.mkdir(parents=True, exist_ok=True)
        source_root = self.overlay if arm=='HG-KDA' else self.repo/'sources'
        launch = ['bash',str(self.project/'tools/evaluation/run-kg-operator-910b.sh'),
            op, 'no-profile' if arm=='kg-no-profile' else 'profile', suffix, '2',
            str(self.batch/self.records[op]['seed'])]
        if not workspace.exists():
            exports = {'KG_SOURCE_ROOT': str(source_root), 'KG_RUN_ROOT': str(run_root),
                'KG_CATALOG_PATH': str(self.batch/'catalog'), 'KG_WALLTIME_PORT':'19659',
                'KG_PROFILE_PORT':'19659', 'KG_MODEL':'deepseek-flash'}
            body = '\n'.join('export '+k+'='+shlex.quote(v) for k,v in exports.items())
            body += '\n' + ' '.join(shlex.quote(x) for x in launch)
            try:
                self.run('kg:'+op+':'+arm, body)
            except RuntimeError:
                # 原生KG终态失败也可能有通过完整测试的轮次，仍允许独立导出复验。
                metadata=workspace/'.kernelgen/run-process.json'
                if not metadata.exists() or not json.loads(metadata.read_text()).get('finished_at'):
                    raise
        metadata=workspace/'.kernelgen/run-process.json'
        if not metadata.exists():
            raise Attention('已有工作区但没有worker身份，不自动重复启动: '+str(workspace))
        if not json.loads(metadata.read_text()).get('finished_at'):
            self.run('wait:'+op+':'+arm, 'export PYTHONPATH='+shlex.quote(str(source_root))+':"$PYTHONPATH"\n'+
                shlex.quote(str(self.root/'runtime/kg-controller/venv/bin/python3'))+' '+
                shlex.quote(str(self.project/'tools/evaluation/wait-kg-worker.py'))+' '+shlex.quote(str(workspace)))
        tools = self.project/'tools/evaluation'
        exported=output/'export';archived=output/'native-evidence'
        for script,destination in [('export-kg-best.py',exported),('archive-kg-evidence.py',archived)]:
            if destination.exists():
                continue
            args=[self.py,str(tools/script),str(workspace)]
            if script.startswith('export'):
                args.append(op)
            args.append(str(destination))
            self.run(script+':'+op+':'+arm, ' '.join(shlex.quote(x) for x in args), budget=600)
        # 记录实际工作区物化的分析角色；不归档Claude全局设置或凭据。
        if arm=='HG-KDA':
            copied=[]
            for path in (p for name in ['kernel-profile-analyzer.md','kernel-knowledge-profile-analyzer.md'] for p in workspace.rglob(name)):
                if path.is_file() and 'HG-KDA 角色扩展' in path.read_text(errors='replace'):
                    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
                    target=output/('loaded-role-'+digest[:12]+'.md');target.write_bytes(data)
                    copied.append({'path':str(path),'sha256':digest,'archived':target.name})
            save(output/'role-load-evidence.json', {'hgkda_roles_found':copied,
                'overlay_manifest':json.loads((self.overlay/'hgkda-overlay.json').read_text())})
        return self.evaluate(op, exported/'main.py', output/'independent', arm, 2)

    def job(self, key, function):
        if (self.runtime/'STOP-AFTER-JOB').exists():
            raise Attention('收到在步骤边界停止的本批标记')
        if self.state['jobs'].get(key,{}).get('status') in ['passed','failed']:
            return
        self.state.update(status='running', current_job=key, updated_at=now())
        self.state['jobs'][key]={'status':'running','started_at':now()};save(self.path,self.state)
        try:
            result=function()
        except Attention:
            raise
        except Exception as error:
            self.idle()
            result={'status':'failed','error':str(error)}
        result['finished_at']=now();self.state['jobs'][key]=result;save(self.path,self.state)
        self.summarize()

    def summarize(self):
        lines=['# 夜间算子体检进展','','自动汇总当前已知结果；Native完整调用计时，不与旧设备kernel成绩混用。','',
               '| 算子 | 固定原版 | KG无profiler | KG原生profiler | HG-KDA |','|---|---|---|---|---|']
        for op in self.records:
            columns=[]
            for arm in ['FlagGems','kg-no-profile','kg-profile','HG-KDA']:
                item=self.state['jobs'].get(('baseline:' if arm=='FlagGems' else 'kg:')+op+('' if arm=='FlagGems' else ':'+arm),{})
                values=item.get('scores',[])
                columns.append('/'.join('%.3f×'%x for x in values) if values else item.get('status','待运行'))
            lines.append('| '+op+' | '+' | '.join(columns)+' |')
        (self.batch/'进展.md').write_text('\n'.join(lines)+'\n')

    def execute(self):
        self.idle()
        if self.plan.get('execution_mode')!='baseline_only' and not self.overlay.exists():
            self.run('build-overlay', ' '.join(shlex.quote(x) for x in [self.py,
                str(self.project/'tools/profiling/hgkda/build-kg-overlay.py'),'--output',str(self.overlay)]),600)
        # 先全量体检，避免夜间只有一个算子的长日志而没有总体状态。
        priority=['value_selecting_reduction_backward','moe_sum_reduce']
        order=[op for op in priority if op in self.records]+[op for op in self.records if op not in priority]
        for op in order:
            self.job('baseline:'+op,lambda op=op:self.baseline(op))
        if self.plan.get('execution_mode')!='baseline_only':
            for op in priority:
                for arm in ['kg-no-profile','kg-profile','HG-KDA']:
                    self.job('kg:'+op+':'+arm,lambda op=op,arm=arm:self.kg(op,arm))
            for op in ['selu_','log10_']:
                self.job('kg:'+op+':HG-KDA',lambda op=op:self.kg(op,'HG-KDA'))
        self.state.update(status='completed',current_job=None,updated_at=now());save(self.path,self.state)
        self.summarize()


if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',default='night-20261021')
    p.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'))
    p.add_argument('--container',default='tle_yy')
    p.add_argument('--wait-for-lock',action='store_true',help='仅在明确排队的补测中等待本项目卡锁')
    a=p.parse_args()
    if not a.batch.replace('-','').replace('_','').isalnum():raise ValueError('非法批次名')
    queue=Queue(a.batch,a.root,a.container)
    with (a.root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if a.wait_for_lock else fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:queue.execute()
        except Exception as error:
            queue.state.update(status='needs_attention',error=str(error),updated_at=now())
            save(queue.path,queue.state);queue.summarize();raise
