"""多kernel调用的真机中间输入快照与目标模拟器回放协调器。"""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys

from ir_campaign import read, save, sha
from ir_collect import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--invocation', required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--isolation-id', required=True)
    args=p.parse_args()
    assert Path(args.isolation_id).name == args.isolation_id
    out=args.run.resolve()/'observations'/args.invocation
    assert not (out/'status.json').exists()
    config=read(out/'config.json');config.update(package=str(args.package.resolve()))
    pair=out/'paired';spec=read(pair/'spec.json')
    assert spec['source_sha256'] == config['source_sha256']
    kernel=read(out/'simulator-selection.json')['kernel']
    record=out/'isolated-replays'/args.isolation_id
    record.mkdir(parents=True,exist_ok=False)
    save(record/'config.json',config)
    environment=dict(os.environ,PYTHONPATH=str(args.package.resolve()/'src'))
    sys.path.insert(0,str(args.package.resolve()/'src'))
    from hgkda.runtime.ir_input import freeze
    frozen,binding=freeze(spec['ir_input']['manifest'],record/'implementation',Path(spec['source']),
        runtime_path=config['ir_capture_runtime'])
    snapshot_spec=dict(spec,source=str(frozen),ir_input=binding)
    save(record/'snapshot-spec.json',snapshot_spec)
    command=[config['device_python'],'-m','hgkda.runtime.launch_replay','--spec',str(record/'snapshot-spec.json'),
        '--kernel-name',kernel,'--output',str(record/'launch-snapshot')]
    shell='source '+shlex.quote(config['sdk']+'/set_env.sh')+'\n'
    values=dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']),OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(args.package.resolve()/'src'),TRITON_CACHE_DIR=str(record/'real-triton-cache'))
    shell+='\n'.join('export '+key+'='+shlex.quote(value) for key,value in values.items())
    shell+='\nexec '+shlex.join(command)
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (record/'snapshot.log').open('x') as stream:
            process=subprocess.run(['docker','exec',config['container'],'bash','-c',shell],
                stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    save(record/'snapshot-process.json',dict(command=command,returncode=process.returncode))
    assert process.returncode == 0
    snapshot=read(record/'launch-snapshot/manifest.json')
    simulator=out/'simulator'
    previous={}
    if simulator.exists():
        archive=out/'simulator-history'/args.isolation_id
        archive.mkdir(parents=True,exist_ok=False)
        previous=dict(previous_execution=read(simulator/'execution.json'),
            previous_request_sha256=sha(simulator/'request.json'))
        simulator.rename(archive/'simulator')
    save(record/'provenance.json',dict(started_at=datetime.now(timezone.utc).isoformat(),
        reason='Explicit later-stage kernel analysis using a real preceding-stage tensor snapshot',
        **previous,
        snapshot_manifest_sha256=sha(record/'launch-snapshot/manifest.json'),
        preceding_real_launches=snapshot['preceding_real_launches'],source_sha256=spec['source_sha256'],
        tool_sha256=sha(__file__),package_manifest_sha256=sha(args.package/'method-manifest.json'),
        scope='Target-kernel simulator replay; whole-call formal results and paired metrics remain authoritative'))
    command=['-m','hgkda.runtime.collect_simulator','--config',str(record/'config.json'),
        '--paired-directory',str(pair),'--output',str(simulator),'--kernel-name',kernel,
        '--kernel-replay',str(record/'launch-snapshot/manifest.json'),'--soc','Ascend910B4',
        '--simulator-library',config['sdk']+'/aarch64-linux/simulator/Ascend910B4/lib',
        '--timeout-minutes','2','--trace-unit','us','--allow-postcapture-exit']
    with (record/'simulator.log').open('x') as stream:
        process=subprocess.run([sys.executable,*command],env=environment,stdout=stream,stderr=subprocess.STDOUT)
    save(record/'simulator-process.json',dict(command=command,returncode=process.returncode))
    summary=summarize(config,simulator)
    save(out/'status.json',dict(status='ir_evidence_collected',source_sha256=config['source_sha256'],
        ir_manifest_sha256=read(out/'input-provenance.json')['manifest_sha256'],reports=len(summary['reports']),
        simulator_returncode=process.returncode,isolated_replay=str(record),
        scope='Real preceding-stage tensors; selected launch only in simulator; complete-call measurements separate'))


if __name__ == '__main__':
    main()
