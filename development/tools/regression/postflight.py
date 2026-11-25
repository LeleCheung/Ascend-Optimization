"""Wait for a fixed regression, audit and summarize evidence, then run pending official-reference qualification probes serially."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--probe-directory',type=Path,required=True)
    parser.add_argument('--container',required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--device-python',required=True)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,required=True)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    repository = Path(read(run/'plan.json')['repository'])
    receipt = run/'postflight-state.json'
    if receipt.exists():
        raise FileExistsError('An existing postflight identity prevents duplicate startup')
    state = dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),status='waiting_for_regression')
    save(receipt,state)
    try:
        while True:
            current = read(run/'state.json')
            if current['status'] == 'completed':
                assert current['completed'] == current['expected']
                break
            if current['status'] == 'requires_attention':
                raise RuntimeError('Original queue needs attention; do not skip unresolved jobs')
            time.sleep(10)
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert (run/'service-stop.json').exists(), 'Wait for the original controller to stop this batch service and release its lock'
            state['status'] = 'auditing'
            save(receipt,state)

            def execute(label,command):
                with (run/(label+'.log')).open('xb') as output:
                    result = subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,cwd=repository)
                if result.returncode:
                    raise RuntimeError(label+'Failed; inspect the corresponding raw log')

            execute('postflight-audit',[sys.executable,str(repository/'tools/regression/audit.py'),
                     '--run-directory',str(run),'--output',str(run/'audit-final.json')])
            execute('postflight-summary',[sys.executable,str(repository/'tools/regression/summarize.py'),
                     '--run-directory',str(run),'--output',str(run/'summary-final.json')])
            args.probe_directory.mkdir(parents=True,exist_ok=False)
            catalog = args.probe_directory/'catalog'
            catalog.mkdir()
            original = (args.catalog/'correctness.jsonl').read_bytes()
            (catalog/'correctness.jsonl').write_bytes(original)
            probe = repository/'tools/operators/expm1_/probe_cann.py'
            snapshot = args.probe_directory/probe.name
            snapshot.write_bytes(probe.read_bytes())
            save(args.probe_directory/'identity.json',dict(
                 checker_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                 catalog_sha256=hashlib.sha256(original).hexdigest(),
                 original_catalog=str(args.catalog),device=args.device,container=args.container,
                 purpose='Probe official CANN Expm1 precision modes after regression; preserve candidates and keep this separate from full acceptance'))
            state['status'] = 'probing_official_reference'
            save(receipt,state)
            for mode in ['default','high_precision']:
                output = args.probe_directory/mode
                command = [args.device_python,str(snapshot),'--mode',mode,'--sdk',str(args.sdk),
                           '--catalog',str(catalog),'--output',str(output),'--physical-device',str(args.device)]
                environment = dict(ASCEND_RT_VISIBLE_DEVICES=str(args.device),PYTHONDONTWRITEBYTECODE='1',
                                   OMP_NUM_THREADS='1',ASCEND_WORK_PATH=str(args.probe_directory/'vendor-work'),
                                   ASCEND_CACHE_PATH=str(args.probe_directory/'vendor-cache'))
                shell = 'source '+shlex.quote(str(args.sdk/'set_env.sh'))+'\n'
                shell += '\n'.join('export '+key+'='+shlex.quote(value) for key,value in environment.items())+'\n'
                shell += 'exec '+shlex.join(command)
                state['mode'] = mode
                save(receipt,state)
                execute('postflight-expm1-'+mode,['docker','exec',args.container,'bash','-c',shell])
                assert (output/'result.json').exists()
            state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat())
            save(receipt,state)
    except Exception as error:
        state.update(status='requires_attention',error=type(error).__name__+': '+str(error))
        save(receipt,state)
        raise


if __name__ == '__main__':
    main()
