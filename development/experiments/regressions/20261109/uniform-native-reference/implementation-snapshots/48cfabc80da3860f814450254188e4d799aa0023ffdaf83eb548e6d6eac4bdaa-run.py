"""串行执行独立 CLI 回归；每项只提交一次完整评测，未知任务不自动重发。"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from hgkda.validation import supplement_status


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--device-python', required=True)
    parser.add_argument('--cann-environment', type=Path, required=True)
    parser.add_argument('--operator', action='append')
    args = parser.parse_args()
    plan = read(args.plan)
    assert not plan['errors'] and len(plan['jobs']) == plan['expected']
    run = args.plan.resolve().parent
    repo = Path(plan['repository'])
    assert sha(repo/'benchmarks/index.json') == plan['index_sha256']
    state_path = run/'state.json'
    state = read(state_path) if state_path.exists() else dict(schema_version='hgkda.regression/v1',
                 started_at=now(), status='prepared', operators={}, expected=plan['expected'])
    server = 'http://127.0.0.1:' + str(args.port)
    compatibility = []
    process = None
    server_pid = None

    def persist():
        state['updated_at'] = now()
        save(state_path, state)

    def call(label, argv, *, timeout=2100):
        log = run/(label+'.log')
        if log.exists():
            raise RuntimeError('操作已有日志，须核验状态后才能恢复：'+label)
        with log.open('xb') as output:
            result = subprocess.run([str(x) for x in argv], stdout=output, stderr=subprocess.STDOUT,
                                    cwd=repo, timeout=timeout)
        if result.returncode:
            raise RuntimeError(label+' 返回 '+str(result.returncode)+': '+log.read_text(errors='replace')[-1800:])

    def cli(label, arguments):
        call(label, [sys.executable, '-m', 'hgkda', *arguments])

    def device_call(label, arguments):
        environment = {'ASCEND_RT_VISIBLE_DEVICES':str(args.device),
                       'PYTHONDONTWRITEBYTECODE':'1', 'OMP_NUM_THREADS':'1',
                       'PYTHONPATH':str(repo/'third_party/FlagGems/src'),
                       'TRITON_CACHE_DIR':str(run/'supplement-triton-cache')}
        shell = 'source '+shlex.quote(str(args.cann_environment))+'\n'
        shell += '\n'.join('export '+key+'='+shlex.quote(value) for key,value in environment.items())+'\n'
        shell += 'exec '+shlex.join([args.device_python,*map(str,arguments)])
        call(label, ['docker','exec',args.container,'bash','-c',shell], timeout=1500)

    def supplement(job, kind):
        spec = job['checks'][kind]
        out = run/'operators'/job['operator']/(kind+'.json')
        assert sha(spec['checker']) == spec['checker_sha256']
        for name,digest in spec['contract_sha256'].items():
            assert sha(Path(spec['leaf'])/name) == digest
        if not out.exists():
            device_call(job['operator']+'-'+kind,[spec['checker'],'--operator',job['operator'],
                        '--source',job['source'],'--leaf',spec['leaf'],'--output',out])
        value = read(out)
        return dict(status=supplement_status(job['operator'],value,job['source_sha256']),
                    passed=value['passed'],total=value['total'],path=str(out),sha256=sha(out),
                    checker_sha256=spec['checker_sha256'])

    def owner_process():
        found = []
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv = path.read_bytes().split(b'\0')
            except OSError:
                continue
            if b'kernelgen_server.server' in argv and str(args.port).encode() in argv:
                if str(run/'profiles').encode() in argv:
                    found.append(int(path.parent.name))
        if len(found) != 1:
            raise RuntimeError('专用服务身份不唯一：'+str(found))
        return found[0]

    with args.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            state['status'] = 'starting'
            persist()
            with socket.socket() as probe:
                assert probe.connect_ex(('127.0.0.1',args.port)) != 0, '端口已占用，禁止接管未知服务'
            for record in plan['compatibility_files']:
                path, archived = Path(record['path']), Path(record['archived'])
                assert path.is_relative_to(repo/'project') and sha(archived) == record['sha256']
                if path.exists():
                    assert sha(path) == record['sha256']
                    continue
                path.parent.mkdir(parents=True,exist_ok=True)
                path.symlink_to(archived)
                compatibility.append(record)
            save(run/'compatibility-links.json',compatibility)
            environment = {'ASCEND_RT_VISIBLE_DEVICES':str(args.device), 'KGS_FLAGGEMS_ROOT':str(repo/'third_party/FlagGems'),
                           'PYTHONPATH':':'.join(str(repo/p) for p in ['sources/kernelgen_server/client','sources/kernelgen_server','sources']),
                           'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost',
                           'TRITON_CACHE_DIR':str(run/'service-triton-cache'), 'PYTHONDONTWRITEBYTECODE':'1'}
            command = [args.device_python,'-u','-m','kernelgen_server.server',
                       '--host','127.0.0.1','--backend','npu','--timing','walltime','--max-workers','1',
                       '--port',str(args.port),'--profile-artifact-root',str(run/'profiles')]
            shell = 'source '+shlex.quote(str(args.cann_environment))+'\n'
            shell += '\n'.join('export '+key+'='+shlex.quote(value) for key,value in environment.items())+'\n'
            shell += 'exec '+shlex.join(command)+'\n'
            script = run/('service-'+str(int(time.time()))+'.sh')
            script.write_text(shell)
            with (run/('service-'+str(int(time.time()))+'.log')).open('xb') as log:
                process = subprocess.Popen(['docker','exec',args.container,'bash',str(script)],
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError('专用服务启动失败')
                try:
                    with opener.open(server+'/status',timeout=2) as response:
                        status = json.load(response)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError('专用服务未就绪')
            server_pid = owner_process()
            save(run/'service-identity.json',dict(pid=server_pid,
                 starttime=Path('/proc/'+str(server_pid)+'/stat').read_text().split()[21],
                 command=command,status=status,device=args.device,container=args.container))
            state['status'] = 'running'
            persist()
            for job in plan['jobs']:
                op = job['operator']
                if args.operator and op not in args.operator:
                    continue
                prior = state['operators'].get(op,{})
                if prior.get('terminal'):
                    continue
                state['current_operator'] = op
                row = state['operators'].setdefault(op,dict(status='running',started_at=now(),
                      source_sha256=job['source_sha256'],previous_cann_score=job['previous_cann_score']))
                persist()
                print('START',op,flush=True)
                root = run/'operators'/op
                try:
                    assert sha(job['source']) == job['source_sha256']
                    if not (root/'gate.json').exists():
                        cli(op+'-gate',['gate','--source',job['source'],'--review',job['review'],'--output',root/'gate.json'])
                    if not (root/'upload.json').exists():
                        cli(op+'-upload',['kgs','upload','--server',server,'--archive',job['archive'],'--output',root/'upload.json'])
                    result_path = root/'evaluation/result.json'
                    if not result_path.exists():
                        if (root/'evaluation').exists():
                            raise RuntimeError('已有提交目录但结果未知；必须查询原 operation_id，禁止重发')
                        cli(op+'-evaluate',['kgs','evaluate','--server',server,'--request',job['request'],
                            '--review',job['review'],'--run-directory',root/'evaluation','--output',root/'evaluate.json'])
                    result = read(result_path)
                    row['evaluation'] = {k:result.get(k) for k in
                        ['status','geo_mean','num_passed','num_workloads','is_hack','server_backend']}
                    row['evaluation'].update(path=str(result_path),sha256=sha(result_path))
                    row['supplementary'] = supplement(job,'supplementary')
                    persist()
                    row['cann_supplementary'] = supplement(job,'cann_supplementary')
                    full = (result.get('status')=='PASSED' and result.get('num_passed')==result.get('num_workloads')
                            and result.get('is_hack') is False)
                    extras = all(row[k]['status']=='passed' for k in ['supplementary','cann_supplementary'])
                    row.update(accepted=bool(full and extras and result.get('geo_mean',0)>.8),
                               status='completed',terminal=True,finished_at=now())
                    print('DONE',op,result.get('geo_mean'),'accepted='+str(row['accepted']),flush=True)
                except Exception as error:
                    row.update(status='requires_review',error=type(error).__name__+': '+str(error),
                               terminal=True,accepted=False,finished_at=now())
                    print('ERROR',op,row['error'],flush=True)
                    if (root/'evaluation/transport-error.json').exists():
                        persist()
                        raise RuntimeError('评测任务结果未知；暂停队列以避免设备任务重叠') from error
                state['completed'] = sum(bool(v.get('terminal')) for v in state['operators'].values())
                state['accepted'] = sum(bool(v.get('accepted')) for v in state['operators'].values())
                persist()
            state.update(status='completed',current_operator=None,finished_at=now())
            persist()
        except Exception as error:
            state.update(status='requires_attention',error=type(error).__name__+': '+str(error))
            persist()
            raise
        finally:
            if server_pid is not None:
                # 只停止本轮明确身份的服务；未知在途任务保留服务及依赖以便查询。
                uncertain = any((run/'operators'/job['operator']/'evaluation/transport-error.json').exists()
                                and not (run/'operators'/job['operator']/'evaluation/result.json').exists()
                                for job in plan['jobs'])
                if not uncertain:
                    assert owner_process() == server_pid
                    os.kill(server_pid,signal.SIGTERM)
                    if process is not None:
                        process.wait(timeout=30)
                    save(run/'service-stop.json',dict(pid=server_pid,stopped_at=now(),scope='本批专用服务'))
                    for record in reversed(compatibility):
                        path = Path(record['path'])
                        if path.is_symlink() and path.resolve()==Path(record['archived']).resolve():
                            path.unlink()
                    save(run/'compatibility-cleanup.json',dict(removed_files=len(compatibility)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
