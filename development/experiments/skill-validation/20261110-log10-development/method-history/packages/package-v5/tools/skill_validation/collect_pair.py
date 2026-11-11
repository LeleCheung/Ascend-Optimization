"""为当前生成候选与冻结CANN参考采集同输入指标；先从设备任务发现实际kernel。"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=read(a.config);package=Path(c['package']);run=Path(c['run_directory'])
    source=a.source.resolve();digest=sha(source);a.output.mkdir(parents=True,exist_ok=False)
    leaf=run/'reference-catalog'
    case=next(json.loads(line) for line in (leaf/'timing.jsonl').read_text().splitlines() if json.loads(line)['name']==a.case)
    spec=a.output/'spec.json'
    save(spec,dict(operator=c['operator'],source=str(source),source_sha256=digest,leaf=str(leaf),case=case,
         reference_kind=c['job']['reference_kind'],entrypoint='run',ir_directory=str(a.output/'actual-ir')))
    environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(c['device']),OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
          PYTHONPATH=str(Path(c['repository'])/'third_party/FlagGems/src'),TRITON_CACHE_DIR=str(a.output/'profile-triton-cache'))
    def device(label,arguments):
        shell='source '+shlex.quote(str(Path(c['sdk'])/'set_env.sh'))+'\n'
        shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())+'\n'
        shell+='exec '+shlex.join(list(map(str,arguments)))
        with (a.output/(label+'.log')).open('xb') as log:
            result=subprocess.run(['docker','exec',c['container'],'bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        assert result.returncode==0,label+': '+(a.output/(label+'.log')).read_text(errors='replace')[-1800:]
    with Path(c['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for side in ['candidate','reference']:
            trace=a.output/(side+'-discovery.json')
            device(side+'-discover',[c['device_python'],package/'tools/skill_validation/discover_kernel.py',
                   '--spec',spec,'--side',side,'--output',trace])
            names=read(trace)['kernels'];assert len(names)==1,'目标路径包含多个kernel，须逐个采集，不能默认选择第一项：'+str(names)
            folder=a.output/side;folder.mkdir()
            application=shlex.join([c['device_python'],str(package/'tools/skill_validation/paired_driver.py'),'--spec',str(spec),'--side',side])
            command=['msprof','op','--application='+application,'--output='+str(folder/'profile'),
                     '--kernel-name='+names[0],'--launch-count=1','--warm-up=5','--aic-metrics=Roofline']
            device(side+'-collect',command)
            basic=list((folder/'profile').rglob('OpBasicInfo.csv'));assert len(basic)==1
            with basic[0].open() as stream:rows=list(csv.DictReader(stream))
            assert len(rows)==1 and rows[0]['Op Name']==names[0]
            save(folder/'collection.json',dict(status='completed',side=side,expected_kernel=names[0],groups='Roofline',
                spec_sha256=sha(spec),command=command,discovery_sha256=sha(trace),
                driver_sha256=sha(package/'tools/skill_validation/paired_driver.py')))
    comparison=a.output/'comparison.json';context=a.output/'context.json'
    subprocess.run([sys.executable,'-m','hgkda','compare','--directory',str(a.output),'--output',str(comparison)],check=True)
    subprocess.run([sys.executable,'-m','hgkda','context','--operator',c['operator'],'--source',str(source),
         '--comparison',str(comparison),'--ir-manifest',str(a.output/'actual-ir/manifest.json'),'--output',str(context)],check=True)
    save(run/'workspace/paired-context-pointer.json',dict(source_sha256=digest,context=str(context),sha256=sha(context)))
    print(json.dumps(dict(status='completed',source_sha256=digest,case=a.case,context=str(context))))


if __name__=='__main__':main()
